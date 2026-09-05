# Runtime assets (never committed)

Place the actual files under this directory:

```
artifacts/
  fashionclip_item_embeddings.pt
  embedding_manifest_v1.json
  polyvore_core7_v2/core7_drop_v2/
    core7_item_metadata_v1_train.jsonl
    core7_item_metadata_v1_valid.jsonl
    core7_item_metadata_v1_test.jsonl
  images/
    <item_id>.jpg              # original bytes from HF
    # OR the original images-*.zip archives; do not supply both for the same item
  retrieval_catalog.jsonl      # generated, never fabricated
```

Download the cache, manifest and metadata from the [provided Drive folder](https://drive.google.com/drive/folders/1mmPithIoJWLQD21pX5kZTtQ37eh7aTS6). Ordinary server deployment does not need a ChatGPT connector. Copy your downloaded files with scp/rsync, or mount an already provisioned directory using `ARTIFACT_HOST_DIR`.

If the existing image archives from the old project are available, put them in `images/`. Otherwise provision from the inspected HF dataset (about 3.74 GB compressed source download; allow additional cache and extracted-image disk space):

```bash
python -m pip install -r scripts/requirements-artifacts.txt
python scripts/download_catalog_images.py \
  --metadata artifacts/polyvore_core7_v2/core7_drop_v2/core7_item_metadata_v1_{train,valid,test}.jsonl \
  --output artifacts/images
```

Then validate and create the production catalog:

```bash
python scripts/prepare_retrieval_catalog.py \
  --artifact-dir artifacts \
  --metadata artifacts/polyvore_core7_v2/core7_drop_v2/core7_item_metadata_v1_{train,valid,test}.jsonl \
  --output artifacts/retrieval_catalog.jsonl
```

The expected catalog is **78,182** verified images, with at least three per canonical category. Any missing image, duplicate image ID, conflicting category, checksum mismatch or unreadable image fails preparation. Never replace missing images with placeholders. Preserve the whole image directory/archives when moving the catalog: paths are relative to `ARTIFACT_DIR`.

Example derived record (an illustration of the schema, not a provided production catalog):

```json
{"item_id":"201869327_1","category":"Top","master_category":"Tank Tops","image_path":"images/201869327_1.jpg"}
```

A ZIP record additionally has `zip_member`. The image API accepts only catalog item IDs and never accepts arbitrary filesystem paths or remote URLs. For source schema, checksums and counts see [ARTIFACT_INSPECTION.md](../docs/ARTIFACT_INSPECTION.md).

Qwen/FashionCLIP weights are downloaded at startup into the mounted HF cache. No model weight is baked into Docker. Preload while network is available if the server will run offline. The supplied embedding manifest does not pin the FashionCLIP commit hash; set `FASHIONCLIP_REVISION` to the original cache encoder revision if known. Do not silently switch encoder models or pretend this missing provenance is verified.
