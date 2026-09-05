import hashlib
import json
import numpy as np
import pytest
import torch
from app.services.catalog import canonical_category, load_cache, safe_path
from app.schemas import Category
from app.errors import AppError


@pytest.mark.parametrize("c", list(Category))
def test_mapping(c):
    assert canonical_category(c.value.upper()) == c


def test_unknown_mapping():
    with pytest.raises(ValueError):
        canonical_category("ACCESSORY")


def test_filter_and_duplicate_fallback(catalog):
    hits = catalog.retrieve(["black shoes", "white shoes", "brown shoes"], np.array([[1, 0, 0]] * 3), "Shoes")
    assert [h["item_id"] for h in hits] == ["Shoes_0", "Shoes_1", "Shoes_2"]
    assert all(h["category"] == "Shoes" for h in hits)


@pytest.mark.parametrize("vectors", [np.zeros((3, 3)), np.full((3, 3), np.nan)])
def test_invalid_vectors(catalog, vectors):
    with pytest.raises(AppError):
        catalog.retrieve(["x"] * 3, vectors, "Shoes")


def test_too_few_candidates(catalog):
    catalog.indices["Shoes"] = catalog.indices["Shoes"][:2]
    with pytest.raises(AppError):
        catalog.retrieve(["x"] * 3, [[1, 0, 0]] * 3, "Shoes")


def test_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../secret")
    with pytest.raises(ValueError):
        safe_path(tmp_path, "/etc/passwd")


def test_real_schema_tiny_cache(tmp_path):
    p = tmp_path / "cache.pt"
    m = tmp_path / "manifest.json"
    tensor = torch.zeros((3, 512), dtype=torch.float16)
    tensor[:, 0] = 1
    torch.save(
        {
            "model_id": "patrickjohncyh/fashion-clip",
            "item_ids": ["a", "b", "c"],
            "embeddings": tensor,
            "normalized": True,
        },
        p,
    )
    manifest = dict(
        model_name_or_version="patrickjohncyh/fashion-clip",
        normalization="l2",
        preprocessing_version="fashionclip-hf-clipprocessor-rgb-v1",
        embedding_version="fashionclip-512-l2-v1",
        embedding_dimension=512,
        dtype="float16",
        item_count=3,
        cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    m.write_text(json.dumps(manifest))
    ids, matrix, _ = load_cache(p, m)
    assert ids == ["a", "b", "c"] and matrix.shape == (3, 512)
    manifest["item_count"] = 4
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_cache(p, m)
    manifest["cache_sha256"] = "0" * 64
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_cache(p, m)
