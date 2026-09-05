#!/usr/bin/env python3
"""Provision actual item images from the inspected HF items Parquet schema."""

import argparse
import io
import json
from pathlib import Path
import re
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
import pyarrow.parquet as pq


def download(metadata, output, revision):
    wanted = set()
    for source in metadata:
        wanted.update(
            json.loads(line)["item_id"] for line in Path(source).read_text().splitlines() if line.strip()
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo = "codewaly/polyvore1000"
    commit = api.dataset_info(repo, revision=revision).sha
    files = sorted(
        p
        for p in api.list_repo_files(repo, repo_type="dataset", revision=commit)
        if p.startswith("items/") and p.endswith(".parquet")
    )
    found = set()
    for name in files:
        local = hf_hub_download(repo, name, repo_type="dataset", revision=commit)
        parquet = pq.ParquetFile(local)
        if not {"item_id", "image"}.issubset(parquet.schema_arrow.names):
            raise ValueError("Unexpected HF items schema")
        for batch in parquet.iter_batches(batch_size=256, columns=["item_id", "image"]):
            for row in batch.to_pylist():
                item = row["item_id"]
                if item not in wanted:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9_-]+", item) or item in found:
                    raise ValueError("Unsafe or duplicate image ID")
                found.add(item)
                image = row["image"]
                if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
                    raise ValueError("Expected embedded image bytes; will not guess remote paths")
                with Image.open(io.BytesIO(image["bytes"])) as im:
                    if im.width * im.height > 20000000:
                        raise ValueError("Oversized source image")
                    im.load()
                    # Preserve original bytes and suffix: cache provenance must not be altered by recompression.
                    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(im.format)
                    if not suffix:
                        raise ValueError("Unsupported source image format")
                target = root / (item + suffix)
                if target.exists() and target.read_bytes() != image["bytes"]:
                    raise ValueError("Existing image differs from source: " + item)
                target.write_bytes(image["bytes"])
        print(f"{name}: {len(found)}/{len(wanted)} images", flush=True)
    if found != wanted:
        raise ValueError(f"{len(wanted - found)} metadata images missing from dataset")
    (root / "source.json").write_text(
        json.dumps({"dataset": repo, "revision": commit, "image_count": len(found)}, indent=2)
    )
    return len(found)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--output", default="artifacts/images")
    p.add_argument("--revision", default="4e012056996e3a60fed0cdbec2d0501426804907")
    a = p.parse_args()
    print("Verified images:", download(a.metadata, a.output, a.revision))
