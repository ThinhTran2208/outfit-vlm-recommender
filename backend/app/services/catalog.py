import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from app.schemas import Category
from app.errors import AppError

CATEGORY_MAP = {c.value.upper(): c for c in Category}


def canonical_category(value):
    try:
        return CATEGORY_MAP[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported source category: {value}") from exc


def safe_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Catalog path escapes artifact root")
    return path


def load_cache(cache_path, manifest_path):
    import torch

    m = json.loads(Path(manifest_path).read_text())
    if m["model_name_or_version"] != "patrickjohncyh/fashion-clip" or m["normalization"] != "l2":
        raise ValueError("Unsupported encoder or normalization")
    if m["preprocessing_version"] != "fashionclip-hf-clipprocessor-rgb-v1":
        raise ValueError("Unsupported preprocessing version")
    if (
        m["embedding_version"] != "fashionclip-512-l2-v1"
        or m["embedding_dimension"] != 512
        or m["dtype"] != "float16"
    ):
        raise ValueError("Unsupported manifest contract")
    with open(cache_path, "rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != m["cache_sha256"]:
            raise ValueError("Cache checksum mismatch")
    c = torch.load(cache_path, map_location="cpu", weights_only=True)
    if c["model_id"] != m["model_name_or_version"] or c["normalized"] is not True:
        raise ValueError("Cache model/normalization mismatch")
    ids = c["item_ids"]
    t = c["embeddings"]
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid/duplicate cache IDs")
    if (
        tuple(t.shape) != (m["item_count"], m["embedding_dimension"])
        or len(ids) != m["item_count"]
        or str(t.dtype) != "torch." + m["dtype"]
    ):
        raise ValueError("Cache shape/dtype mismatch")
    a = t.float().numpy()
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.isfinite(a).all() or np.any(np.abs(norms - 1) > 0.02):
        raise ValueError("Invalid embedding norms")
    return ids, a / norms, m


class Catalog:
    def __init__(self, ids, embeddings, records, root):
        self.root = Path(root).resolve()
        self.records = {}
        id_rows = {item: i for i, item in enumerate(ids)}
        indices = {c: [] for c in Category}
        for row in records:
            item = row["item_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item) or item in self.records or item not in id_rows:
                raise ValueError("Invalid, duplicate, or unembedded catalog item")
            cat = Category(row["category"])
            path = safe_path(self.root, row["image_path"])
            if not path.is_file():
                raise ValueError(f"Missing image/archive for {item}")
            if row.get("zip_member"):
                member = row["zip_member"]
                if member.startswith("/") or ".." in Path(member).parts:
                    raise ValueError("Unsafe ZIP member")
            self.records[item] = dict(row)
            indices[cat].append(id_rows[item])
        self.ids = np.array(ids)
        self.matrix = np.asarray(embeddings, dtype=np.float32)
        self.indices = {cat: np.array(rows, dtype=np.int64) for cat, rows in indices.items()}
        if any(len(rows) < 3 for rows in self.indices.values()):
            raise ValueError("At least three real items per canonical category are required")

    @classmethod
    def load(cls, settings):
        ids, matrix, manifest = load_cache(
            settings.fashionclip_embeddings_path, settings.embedding_manifest_path
        )
        records = [
            json.loads(line)
            for line in settings.retrieval_catalog_path.read_text().splitlines()
            if line.strip()
        ]
        catalog = cls(ids, matrix, records, settings.artifact_dir)
        # Preparation validates all images. Startup verifies ZIP member indexes and file bounds.
        grouped = {}
        for row in records:
            if row.get("zip_member"):
                grouped.setdefault(row["image_path"], []).append(row["zip_member"])
        for archive, members in grouped.items():
            with zipfile.ZipFile(safe_path(catalog.root, archive)) as z:
                for member in members:
                    info = z.getinfo(member)
                    if info.file_size <= 0 or info.file_size > 20 * 1024 * 1024:
                        raise ValueError("Invalid catalog image size")
        return catalog, manifest

    def image_bytes(self, item):
        row = self.records[item]
        p = safe_path(self.root, row["image_path"])
        if row.get("zip_member"):
            with zipfile.ZipFile(p) as z:
                info = z.getinfo(row["zip_member"])
                if info.file_size > 20 * 1024 * 1024:
                    raise ValueError("Oversized catalog image")
                return z.read(info)
        if p.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Oversized catalog image")
        return p.read_bytes()

    def image(self, item):
        with Image.open(io.BytesIO(self.image_bytes(item))) as im:
            if im.width * im.height > 20000000:
                raise ValueError("Oversized catalog image dimensions")
            im.load()
            return im.convert("RGB")

    def retrieve(self, queries, vectors, category):
        rows = self.indices[category]
        q = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        if q.shape != (3, self.matrix.shape[1]) or not np.isfinite(q).all() or np.any(norms <= 1e-12):
            raise AppError("RETRIEVAL_ERROR", "Không thể tìm đủ ba món phù hợp.")
        scores = (q / norms) @ self.matrix[rows].T
        chosen = set()
        results = []
        for rank, similarity in enumerate(scores, 1):
            # Stable sorting gives reproducible ties in original cache order.
            for local in np.argsort(-similarity, kind="stable"):
                item = str(self.ids[rows[local]])
                if item not in chosen:
                    chosen.add(item)
                    results.append(
                        dict(
                            rank=rank,
                            item_id=item,
                            category=category,
                            query_en=queries[rank - 1],
                            cosine_similarity=float(similarity[local]),
                        )
                    )
                    break
        if len(results) != 3:
            raise AppError("RETRIEVAL_ERROR", "Không thể tìm đủ ba món phù hợp.")
        return results
