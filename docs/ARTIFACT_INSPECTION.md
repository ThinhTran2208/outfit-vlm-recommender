# Inspected inputs — 2026-09-05

Reference repository inspected at `e9a7f30119448970608cf9bbdb529053c073145c` on main. Relevant files: `src/vlm/qwen_backend_v2.py`, `configs/vlm_qwen3_vl_4b_instruct_v2.json`, `src/detection/fashionclip.py`, `src/recommendation/metadata.py`, `src/recommendation/zip_artifacts.py`, `src/recommendation/zip_images.py`, Dockerfiles and runtime requirements.

The working reference specifies **Qwen/Qwen3-VL-4B-Instruct**. This implementation preserves the model identifier, processor/chat-template approach, `process_vision_info(image_patch_size=16)`, `do_resize=False`, deterministic generation and generated-token slicing. It adds safe automatic BF16/FP16 selection and a generation deadline. Reference Transformers-v4 code has been adapted for the pinned patched runtime; CLIP projected features also support its `pooler_output` return value. No detector/scorer/calibration/LOO code is imported.

## Google Drive

[Artifact folder](https://drive.google.com/drive/folders/1mmPithIoJWLQD21pX5kZTtQ37eh7aTS6) and its direct Core-7 subfolders were listed through Drive. The actual cache and all three metadata files were downloaded and inspected, not inferred from examples.

Manifest:

```json
{
  "embedding_version": "fashionclip-512-l2-v1",
  "model_name_or_version": "patrickjohncyh/fashion-clip",
  "preprocessing_version": "fashionclip-hf-clipprocessor-rgb-v1",
  "embedding_dimension": 512,
  "normalization": "l2",
  "dtype": "float16",
  "item_count": 142480,
  "cache_sha256": "5bd4f311b460c5051906cf26f86701e26df157a4ea44d0da0a85e08bededb680"
}
```

Actual `.pt`: 148,893,109 bytes; SHA-256 verified against manifest; safe `torch.load(weights_only=True)` returns keys `model_id`, `item_ids`, `embeddings`, `normalized`. `item_ids` is a list of 142,480 strings; tensor shape `[142480,512]`, `torch.float16`; `normalized=True`; model ID matches manifest. Observed L2 norms rounded to four decimal places: 0.9999–1.0001.

Metadata sample:

```json
{"item_metadata_version":"core7-item-metadata-v1","category_mapping_version":"core7-v2","split":"train","item_id":"201869327_1","source_kit_id":"201869327","slot_index":1,"master_category":"Tank Tops","coarse_category":"TOP"}
```

| Split | Rows |
|---|---:|
| train | 64,032 |
| valid | 4,620 |
| test | 9,530 |
| unique total | 78,182 |

All 78,182 unique metadata IDs exist in the embedding cache, with no category conflicts. Metadata contains no image path/URL. Remaining 64,298 embeddings are excluded because the inspected Core-7 metadata does not establish their category.

| Source category | API category | Candidates with metadata |
|---|---|---:|
| TOP | Top | 14,340 |
| BOTTOM | Bottom | 13,711 |
| OUTERWEAR | Outerwear | 8,401 |
| SHOES | Shoes | 19,277 |
| BAG | Bag | 15,246 |
| DRESS | Dress | 4,978 |
| HAT | Hat | 2,229 |

## Real image mapping

The old repository documents image ZIP archives containing item-ID-named JPGs. The preparation script supports these ZIPs without extraction and also local item-ID-named files.

The [HF source](https://huggingface.co/datasets/codewaly/polyvore1000) `items` configuration contains `item_id`, `master_category`, `product_name`, `price`, `image`, `release_date`, `dominant_color`. Its item splits have 114,806 / 9,070 / 18,604 rows. The dataset API and a visible row were inspected: valid row 0 is `216778558_1`, whose real image is 301×400 and whose ID matches Core-7 metadata. We do not use the temporary viewer image URL in production.

`download_catalog_images.py` uses embedded Parquet image bytes at pinned dataset revision `4e012056996e3a60fed0cdbec2d0501426804907`, keeps original bytes, checks IDs, and writes only Core-7 items. It requires complete coverage. `prepare_retrieval_catalog.py` verifies cache hash, joins categories and images, decodes every image offline, and atomically writes the derived catalog. It never edits source metadata.

See VALIDATION.md for the extent of provisioning and real-inference verification completed in the authoring environment. Dataset catalog entries are reference images, not a guarantee that the products are currently for sale.
