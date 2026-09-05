import io
import json
import numpy as np
import pytest
from PIL import Image
from app.config import Settings
from app.schemas import Category, Facet
from app.services.catalog import Catalog
from app.services.models import StructuredVLM
from app.services.pipeline import Pipeline


@pytest.fixture
def analysis():
    garments = [
        dict(garment_id=f"g{i}", category=c, name_en=n, name_vi=v, description_en=n)
        for i, c, n, v in [
            (1, "Top", "white shirt", "áo trắng"),
            (2, "Bottom", "blue jeans", "quần jeans xanh"),
            (3, "Shoes", "red sneakers", "giày đỏ"),
        ]
    ]
    return dict(
        status="ok",
        scene=dict(valid=True, error_code=None, message_vi=None),
        garments=garments,
        counted_item_count=3,
        rubric={f.value: dict(score=16, reason_vi="Các món phối hợp hài hòa.") for f in Facet},
        aesthetic_score=80,
        problematic_item={k: v for k, v in garments[2].items() if k != "description_en"}
        | {"reason_vi": "Có thể thử giày trung tính."},
        replacement_mode="improve",
        replacement_queries_en=[
            "black leather loafers",
            "white minimalist sneakers",
            "brown suede derby shoes",
        ],
    )


@pytest.fixture
def catalog(tmp_path):
    ids = []
    matrix = []
    records = []
    for c in Category:
        for i in range(3):
            item = f"{c.value}_{i}"
            ids.append(item)
            matrix.append([1.0, i / 10, 0.0])
            p = tmp_path / (item + ".jpg")
            Image.new("RGB", (10, 10), (i * 70, 20, 30)).save(p)
            records.append(dict(item_id=item, category=c.value, image_path=p.name))
    m = np.array(matrix)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return Catalog(ids, m, records, tmp_path)


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, system, images, context, deadline):
        self.calls.append((system, images, context))
        value = self.outputs.pop(0)
        return value if isinstance(value, str) else json.dumps(value)


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, queries):
        self.calls.append(queries)
        return np.array([[1.0, 0, 0]] * 3)


@pytest.fixture
def make_pipeline(catalog):
    def make(a, settings=None):
        reasons = {
            "recommendations": [
                dict(
                    rank=i + 1,
                    item_id=f"Shoes_{i}",
                    display_name_en="Shoes",
                    display_name_vi="Giày",
                    reason_facets=["color_harmony"],
                    reason_vi="Màu sắc phù hợp với áo.",
                )
                for i in range(3)
            ]
        }
        backend = FakeBackend([a, reasons])
        encoder = FakeEncoder()
        return (
            Pipeline(StructuredVLM(backend), encoder, catalog, settings or Settings(_env_file=None)),
            backend,
            encoder,
        )

    return make


@pytest.fixture
def jpeg():
    b = io.BytesIO()
    Image.new("RGB", (30, 40), "white").save(b, "JPEG")
    return b.getvalue()
