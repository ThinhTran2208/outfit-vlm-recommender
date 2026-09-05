from copy import deepcopy
import json
import time
import pytest
from pydantic import ValidationError
from app.schemas import Analysis, Explanations, Category, validate_queries
from app.services.models import StructuredVLM, parse_json
from app.errors import AppError
from conftest import FakeBackend


def test_valid(analysis):
    a = Analysis.model_validate(analysis)
    assert a.rubric.total == 80 and len(a.replacement_queries_en) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.update(aesthetic_score=79),
        lambda a: a["rubric"]["color_harmony"].update(score=21),
        lambda a: a["rubric"]["color_harmony"].update(score=-1),
        lambda a: a["rubric"]["color_harmony"].update(score=16.0),
        lambda a: a["problematic_item"].update(garment_id="g9"),
        lambda a: a["problematic_item"].update(category="Bag"),
        lambda a: a.update(replacement_queries_en=["black shoes"] * 3),
        lambda a: a.update(replacement_queries_en=["black shoes", "white sneakers"]),
        lambda a: a.update(replacement_queries_en=["black shoes", "white sneakers", "blue shirt"]),
        lambda a: a.update(counted_item_count=2),
        lambda a: a["garments"][0].update(category="Glasses"),
        lambda a: a["garments"][0].update(garment_id="g2"),
    ],
)
def test_invalid_contract(analysis, mutation):
    mutation(analysis)
    with pytest.raises(ValidationError):
        Analysis.model_validate(analysis)


@pytest.mark.parametrize("code", ["MULTIPLE_PEOPLE", "MANNEQUIN_OR_BACKGROUND_OUTFIT", "AMBIGUOUS_SCENE"])
def test_rejected_cannot_score(analysis, code):
    analysis.update(status="rejected", scene={"valid": False, "error_code": code})
    with pytest.raises(ValidationError):
        Analysis.model_validate(analysis)


@pytest.mark.parametrize(
    "category,query",
    [
        (Category.TOP, "white cotton shirt"),
        (Category.BOTTOM, "black tailored trousers"),
        (Category.OUTERWEAR, "navy wool coat"),
        (Category.SHOES, "white leather sneakers"),
        (Category.BAG, "small leather bag"),
        (Category.DRESS, "long floral dress"),
        (Category.HAT, "black bucket hat"),
    ],
)
def test_query_category(category, query):
    validate_queries([query, "simple " + query, "classic " + query], category)
    wrong = Category.TOP if category != Category.TOP else Category.HAT
    with pytest.raises(ValueError):
        validate_queries([query, "simple " + query, "classic " + query], wrong)


def test_safe_json_and_retry(analysis):
    assert parse_json("```json\n" + json.dumps(analysis) + "\n```")["status"] == "ok"
    assert parse_json("Here is JSON: " + json.dumps(analysis))["status"] == "ok"
    backend = FakeBackend(["broken", analysis])
    v = StructuredVLM(backend)
    assert v.call("outfit_analysis_v1.txt", [], {}, Analysis, time.monotonic() + 10).status == "ok"
    assert len(backend.calls) == 2 and "RETRY" in backend.calls[1][0]


def test_malformed_fails_after_one_retry():
    backend = FakeBackend(["{", 'eval("bad")'])
    v = StructuredVLM(backend)
    with pytest.raises(AppError, match="VLM_OUTPUT_ERROR"):
        v.call("outfit_analysis_v1.txt", [], {}, Analysis, time.monotonic() + 10)
    assert len(backend.calls) == 2


def test_explanation_contract():
    reasons = {
        "recommendations": [
            dict(
                rank=i,
                item_id=str(i),
                display_name_en="shoe",
                display_name_vi="giày",
                reason_vi="Hài hòa.",
                reason_facets=["color_harmony"],
            )
            for i in range(1, 4)
        ]
    }
    Explanations.model_validate(reasons)
    for change in ("facet", "count", "id", "rank"):
        r = deepcopy(reasons)
        if change == "facet":
            r["recommendations"][0]["reason_facets"] = ["price"]
        if change == "count":
            r["recommendations"].pop()
        if change == "id":
            r["recommendations"][0]["item_id"] = "2"
        if change == "rank":
            r["recommendations"][0]["rank"] = 3
        with pytest.raises(ValidationError):
            Explanations.model_validate(r)
