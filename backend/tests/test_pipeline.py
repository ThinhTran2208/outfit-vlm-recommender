import time
import pytest
from PIL import Image
from app.errors import AppError


def test_complete_pipeline(analysis, make_pipeline):
    p, b, e = make_pipeline(analysis)
    with Image.new("RGB", (20, 20)) as im:
        result = p.analyze(im, "test", time.monotonic() + 10, {})
        assert b.calls[1][1][0][1] is im
    assert result.score.total == 80
    assert [r.item_id for r in result.recommendations] == ["Shoes_0", "Shoes_1", "Shoes_2"]
    assert len(b.calls) == 2 and len(b.calls[1][1]) == 4
    assert len(e.calls) == 1 and len(e.calls[0]) == 3
    assert all(r.category == "Shoes" for r in result.recommendations)


@pytest.mark.parametrize(
    "code", [None, "MULTIPLE_PEOPLE", "MANNEQUIN_OR_BACKGROUND_OUTFIT", "AMBIGUOUS_SCENE"]
)
def test_rejected_short_circuit(analysis, make_pipeline, code):
    a = {k: analysis[k] for k in ["garments", "scene", "counted_item_count"]}
    a["status"] = "rejected"
    if code:
        a["scene"] = {"valid": False, "error_code": code}
    else:
        a["garments"] = a["garments"][:2]
        a["counted_item_count"] = 2
    p, b, e = make_pipeline(a)
    with Image.new("RGB", (10, 10)) as im:
        r = p.analyze(im, "test", time.monotonic() + 10, {})
    assert r.error_code == (code or "NOT_ENOUGH_ITEMS")
    assert len(b.calls) == 1 and not e.calls
    assert "score" not in r.model_dump() and "recommendations" not in r.model_dump()


def test_dress_counts_once(analysis, make_pipeline):
    a = {
        "status": "rejected",
        "scene": analysis["scene"],
        "garments": [
            dict(garment_id="g1", category="Dress", name_en="dress", name_vi="váy", description_en="dress"),
            analysis["garments"][2],
        ],
        "counted_item_count": 2,
    }
    p, _, e = make_pipeline(a)
    with Image.new("RGB", (10, 10)) as im:
        r = p.analyze(im, "x", time.monotonic() + 10, {})
    assert r.error_code == "NOT_ENOUGH_ITEMS" and not e.calls


def test_high_score_threshold(analysis, make_pipeline):
    for value in analysis["rubric"].values():
        value["score"] = 17
    analysis.update(aesthetic_score=85, replacement_mode="similar_alternative")
    p, _, _ = make_pipeline(analysis)
    with Image.new("RGB", (10, 10)) as im:
        r = p.analyze(im, "x", time.monotonic() + 10, {})
    assert r.replacement_mode == "similar_alternative" and "biến thể tương tự" in r.commentary_vi


def test_wrong_mode_retried(analysis, make_pipeline):
    analysis["replacement_mode"] = "similar_alternative"
    p, b, _ = make_pipeline(analysis)
    b.outputs = [analysis, analysis]
    with pytest.raises(AppError, match="VLM_OUTPUT_ERROR"):
        p.analyze(Image.new("RGB", (10, 10)), "x", time.monotonic() + 10, {})


def test_grounding_id_mismatch(analysis, make_pipeline):
    p, b, _ = make_pipeline(analysis)
    reasons = b.outputs[1]
    reasons["recommendations"][0]["item_id"] = "not-retrieved"
    b.outputs.append(reasons)
    with pytest.raises(AppError, match="VLM_OUTPUT_ERROR"):
        p.analyze(Image.new("RGB", (10, 10)), "x", time.monotonic() + 10, {})
