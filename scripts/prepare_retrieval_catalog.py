#!/usr/bin/env python3
"""Join observed Core-7 metadata to verified real images; never alter source files."""

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from PIL import Image
from app.services.catalog import canonical_category, load_cache, safe_path
from app.schemas import Category


def prepare(root, metadata, output):
    root = Path(root).resolve()
    ids, _, _ = load_cache(root / "fashionclip_item_embeddings.pt", root / "embedding_manifest_v1.json")
    ids = set(ids)
    records = {}
    for source in metadata:
        with open(source) as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if (
                    raw["item_metadata_version"] != "core7-item-metadata-v1"
                    or raw["category_mapping_version"] != "core7-v2"
                ):
                    raise ValueError("Unexpected metadata version")
                item = raw["item_id"]
                cat = canonical_category(raw["coarse_category"])
                if item not in ids:
                    raise ValueError("Metadata missing cache embedding: " + item)
                if item in records and records[item]["category"] != cat:
                    raise ValueError("Conflicting category: " + item)
                records[item] = {"item_id": item, "category": cat, "master_category": raw["master_category"]}
    images = {}

    def add(item, path, member=None):
        if item not in records:
            return
        if item in images:
            raise ValueError("Duplicate source image: " + item)
        safe_path(root, str(path.relative_to(root)))
        images[item] = {"image_path": str(path.relative_to(root))}
        if member:
            images[item]["zip_member"] = member

    for path in sorted((root / "images").rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            add(path.stem, path)
    for path in sorted((root / "images").glob("*.zip")):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir() and Path(info.filename).suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }:
                    if ".." in Path(info.filename).parts or info.filename.startswith("/"):
                        raise ValueError("Unsafe ZIP member")
                    add(Path(info.filename).stem, path, info.filename)
    missing = set(records) - set(images)
    if missing:
        raise ValueError(f"Missing {len(missing)} Core-7 images; examples: {sorted(missing)[:5]}")
    # Decode every included image once offline, so production does not serve placeholders.
    archives = {}
    verified = []
    try:
        for item in sorted(records):
            ref = images[item]
            path = safe_path(root, ref["image_path"])
            if "zip_member" in ref:
                if path not in archives:
                    archives[path] = zipfile.ZipFile(path)
                z = archives[path]
                info = z.getinfo(ref["zip_member"])
                if info.file_size > 20 * 1024 * 1024:
                    raise ValueError("Oversized image: " + item)
                data = z.read(info)
            else:
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError("Oversized image: " + item)
                data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                if im.width * im.height > 20000000:
                    raise ValueError("Oversized image dimensions: " + item)
                im.load()
            verified.append({**records[item], **ref})
    finally:
        for z in archives.values():
            z.close()
    counts = Counter(r["category"] for r in verified)
    if any(counts[c] < 3 for c in Category):
        raise ValueError("Not enough candidates in every category")
    output = Path(output)
    tmp = output.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in verified))
    tmp.replace(output)
    return {"catalog_items": len(verified), "category_counts": dict(counts), "all_images_decoded": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--output", default="artifacts/retrieval_catalog.jsonl")
    args = p.parse_args()
    print(json.dumps(prepare(args.artifact_dir, args.metadata, args.output), indent=2))
