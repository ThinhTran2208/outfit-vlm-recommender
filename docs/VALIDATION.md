# Validation record — 2026-09-05

## Completed locally

- **59 CPU tests passed** after the final runtime dependency updates (PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, Transformers 5.16.1, FastAPI 0.141.1, Starlette 1.6.0). Two test-client deprecation warnings remain; no test failures.
- Ruff lint and formatter checks passed.
- React TypeScript check and Vite 6.4.3 production build passed, including the later screenshot-reference styling update.
- npm audit: zero reported vulnerabilities in the frontend dependency tree at the time checked.
- Docker Compose v2.39.2 `config --quiet` passed with `.env.example` copied to the ignored `.env`.
- Actual Drive `.pt` hash/schema/shape/dtype/normalization verified; all three metadata files read completely; 78,182 unique metadata IDs join to cached embeddings without conflicts.
- Downloaded **all 78,182 Core-7 source images**, preserving original HF image bytes at pinned dataset revision. Image provisioning script completed successfully.
- Ran `prepare_retrieval_catalog.py` on the actual complete inputs: **78,182 catalog records; every image decoded successfully**. Counts: Top 14,340; Bottom 13,711; Outerwear 8,401; Shoes 19,277; Bag 15,246; Dress 4,978; Hat 2,229.
- Model API imports verified without Qwen weights. A real tiny randomly initialized CLIP test verifies the installed projected-text feature return API and L2-normalized batch output; this is not verification of trained FashionCLIP quality.

The test suite covers successful and rejected pipelines, fewer than three garments, dress/shoe counting contract, scene codes, strict rubric sums/ranges, selected-item membership, three queries and category noun constraints, same-category retrieval, duplicate top-1 fallback, exact unique candidates, malformed JSON retry/failure, grounded explanation IDs/facets, uploads, image serving, readiness, and timeout admission safety.

## Full-catalog CPU retrieval probe

Production loader opened the complete real catalog in 6.19 seconds in this environment. For each of seven categories, three repeated cached-image vectors were used as probes (not trained text queries). Every result had three unique same-category item IDs and all 21 returned images decoded successfully. One-shot category search times ranged from 2.174 to 116.561 ms; these are smoke measurements, not a production latency benchmark.

## Not verified

- **Real Qwen GPU inference: not run.** `torch.cuda.is_available()` is false in this environment. No GPU model quality, latency, VRAM or full real-model end-to-end claim is made.
- Trained FashionCLIP text inference verification was interrupted by an automatic approval rejection due to the session usage limit. The adapter and actual cache/catalog are tested separately; this is not a successful trained-encoder smoke result.
- Docker daemon is unavailable locally. Compose parsing passed; container builds/startup are covered only to the extent reported by GitHub CI. Do not equate Compose parsing with a verified GPU container run.
- Browser screenshot/mobile QA could not run: the cloud browser returned `ERR_BLOCKED_BY_CLIENT` for the supervised internal preview. No screenshot of the new app is fabricated. Two user-provided screenshots guided the later visual implementation; they show the prior app, not proof of the new app's runtime.
- No GPU server session/deployment was provided. The URL in the user screenshot shows an earlier app; this change has not replaced it.

## Server acceptance

Follow [DEPLOYMENT.md](DEPLOYMENT.md), then run:

```bash
python scripts/gpu_smoke.py --url http://localhost:57260 --image /absolute/path/outfit.jpg --expect ok
python scripts/gpu_smoke.py --url http://localhost:57260 --image /absolute/path/two_people.jpg --expect rejected
python scripts/gpu_smoke.py --url http://localhost:57260 --image /absolute/path/dress_and_shoes.jpg --expect rejected
```

Manually inspect scene decisions, garment identity, query diversity and each explanation against the four images. Verify desktop/mobile layout and upload/loading/error states in a normal browser. Record GPU and runtime revisions with results. Passing schema tests alone does not establish visual accuracy or aesthetic agreement with people.
