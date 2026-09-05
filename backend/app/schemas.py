from enum import StrEnum
from typing import Annotated, Literal
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Category(StrEnum):
    TOP = "Top"
    BOTTOM = "Bottom"
    OUTERWEAR = "Outerwear"
    SHOES = "Shoes"
    BAG = "Bag"
    DRESS = "Dress"
    HAT = "Hat"


class Facet(StrEnum):
    COLOR = "color_harmony"
    STYLE = "style_coherence"
    SHAPE = "silhouette_proportion"
    OCCASION = "formality_occasion_coherence"
    OVERALL = "overall_styling"


Text = Annotated[str, Field(min_length=1, max_length=1200)]


class Garment(Strict):
    garment_id: Annotated[str, Field(pattern=r"^g[1-9][0-9]*$")]
    category: Category
    name_en: Text
    name_vi: Text
    description_en: Text


class Problem(Strict):
    garment_id: str
    category: Category
    name_en: Text
    name_vi: Text
    reason_vi: Text


class Component(Strict):
    score: Annotated[int, Field(strict=True, ge=0, le=20)]
    reason_vi: Text


class Rubric(Strict):
    color_harmony: Component
    style_coherence: Component
    silhouette_proportion: Component
    formality_occasion_coherence: Component
    overall_styling: Component

    @property
    def total(self):
        return sum(getattr(self, k).score for k in type(self).model_fields)


class Scene(Strict):
    valid: bool
    error_code: Literal["MULTIPLE_PEOPLE", "MANNEQUIN_OR_BACKGROUND_OUTFIT", "AMBIGUOUS_SCENE"] | None = None
    message_vi: str | None = None


# Conservative, explicit garment noun vocabulary; unknown wording is retried, never silently recategorized.
QUERY_NOUNS = {
    Category.TOP: r"\b(t-?shirts?|shirts?|blouses?|tops?|sweaters?|sweatshirts?|hoodies?|tank tops?|polos?)\b",
    Category.BOTTOM: r"\b(pants|trousers|jeans|skirts?|shorts|leggings|chinos|culottes)\b",
    Category.OUTERWEAR: r"\b(jackets?|coats?|blazers?|cardigans?|parkas?|vests?)\b",
    Category.SHOES: r"\b(shoes|sneakers|loafers|boots|sandals|heels|flats|derbies|derby shoes|oxfords|pumps)\b",
    Category.BAG: r"\b(bags?|handbags?|backpacks?|totes?|clutches|clutch|satchels?)\b",
    Category.DRESS: r"\b(dresses|dress|gowns?|jumpsuits?|rompers?)\b",
    Category.HAT: r"\b(hats?|caps?|beanies|beanie|berets?)\b",
}


def validate_queries(queries, category):
    if len(queries) != 3 or len({q.casefold() for q in queries}) != 3:
        raise ValueError("Exactly three distinct queries required")
    for q in queries:
        if not 2 <= len(q.split()) <= 18 or not q.isascii() or not re.fullmatch(r"[A-Za-z][A-Za-z '\-]*", q):
            raise ValueError("Use concise English garment descriptions")
        hits = {c for c, pattern in QUERY_NOUNS.items() if re.search(pattern, q.lower())}
        if hits != {category}:
            raise ValueError("Query must name only the selected garment category")


class Analysis(Strict):
    status: Literal["ok", "rejected"]
    scene: Scene
    garments: Annotated[list[Garment], Field(max_length=20)]
    counted_item_count: Annotated[int, Field(strict=True, ge=0, le=20)]
    rubric: Rubric | None = None
    aesthetic_score: Annotated[int, Field(strict=True, ge=0, le=100)] | None = None
    problematic_item: Problem | None = None
    replacement_mode: Literal["improve", "similar_alternative"] | None = None
    replacement_queries_en: list[str] | None = None

    @model_validator(mode="after")
    def contract(self):
        if self.counted_item_count != len(self.garments):
            raise ValueError("Count must equal garments length")
        if len({g.garment_id for g in self.garments}) != len(self.garments):
            raise ValueError("Duplicate garment IDs")
        if sum(g.category == Category.SHOES for g in self.garments) > 1:
            raise ValueError("One pair of shoes counts once")
        if self.scene.valid == (self.scene.error_code is not None):
            raise ValueError("Scene validity/error mismatch")
        valid = self.scene.valid and self.counted_item_count >= 3
        if (self.status == "ok") != valid:
            raise ValueError("Invalid scenes or fewer than three garments must be rejected")
        fields = [
            self.rubric,
            self.aesthetic_score,
            self.problematic_item,
            self.replacement_mode,
            self.replacement_queries_en,
        ]
        if not valid:
            if any(v is not None for v in fields):
                raise ValueError("Rejected analysis cannot contain scoring or recommendations")
            return self
        if any(v is None for v in fields):
            raise ValueError("Missing successful analysis fields")
        if self.aesthetic_score != self.rubric.total:
            raise ValueError("Score is not rubric sum")
        p = self.problematic_item
        g = next((g for g in self.garments if g.garment_id == p.garment_id), None)
        if g is None or (g.category, g.name_en, g.name_vi) != (p.category, p.name_en, p.name_vi):
            raise ValueError("Problematic item must match a real garment")
        validate_queries(self.replacement_queries_en, p.category)
        return self


class Reason(Strict):
    rank: Annotated[int, Field(strict=True, ge=1, le=3)]
    item_id: Text
    display_name_en: Text
    display_name_vi: Text
    reason_facets: Annotated[list[Facet], Field(min_length=1, max_length=5)]
    reason_vi: Text


class Explanations(Strict):
    recommendations: Annotated[list[Reason], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def unique(self):
        if [r.rank for r in self.recommendations] != [1, 2, 3] or len(
            {r.item_id for r in self.recommendations}
        ) != 3:
            raise ValueError("Three ordered unique recommendations required")
        return self


class Recommendation(Reason):
    category: Category
    query_en: str
    image_url: str


class Score(Strict):
    total: int
    dimensions: Rubric


class Success(Strict):
    status: Literal["ok"] = "ok"
    garments: list[Garment]
    score: Score
    problematic_item: Problem
    replacement_mode: Literal["improve", "similar_alternative"]
    recommendations: Annotated[list[Recommendation], Field(min_length=3, max_length=3)]
    commentary_vi: str


class Rejected(Strict):
    status: Literal["rejected"] = "rejected"
    error_code: str
    message_vi: str
    counted_item_count: int
    garments: list[Garment]


class Error(Strict):
    status: Literal["error"] = "error"
    error_code: str
    message_vi: str
    request_id: str | None = None
