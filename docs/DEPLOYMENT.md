# Deploy on Linux with NVIDIA GPU

The deliverable is a repository ready to provision and run on your GPU server. No server IP/access was supplied and no deployment was performed from the authoring environment.

## Requirements

- Linux x86-64, Docker Engine with Compose v2 and NVIDIA Container Toolkit.
- NVIDIA driver compatible with CUDA 12.4 containers; verify `nvidia-smi` and a GPU container before deployment.
- Start with a 16–24 GB VRAM GPU and at least 16 GB system RAM; these are planning estimates, not measured minima. FP16 4B weights alone are about 8 GB, with additional vision/KV/activation overhead, particularly for four-image explanation.
- Allow roughly 25 GB free disk for image sources, extracted images, HF cache and build layers; actual use depends on caching. Internet access for first model/source download, or preloaded caches.

## Clone and provision

```bash
git clone --branch feat/end-to-end-vlm-recommender https://github.com/ThinhTran2208/outfit-vlm-recommender.git
cd outfit-vlm-recommender
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt -r scripts/requirements-artifacts.txt
```

Copy the original Drive cache, manifest and three metadata files into the exact structure in [artifacts/README.md](../artifacts/README.md). Provisioning uses CPU only; the Docker backend installs CUDA PyTorch separately.

If you do not already have image ZIPs:

```bash
python scripts/download_catalog_images.py \
  --metadata artifacts/polyvore_core7_v2/core7_drop_v2/core7_item_metadata_v1_{train,valid,test}.jsonl \
  --output artifacts/images
```

For either image source, build the catalog:

```bash
python scripts/prepare_retrieval_catalog.py \
  --artifact-dir artifacts \
  --metadata artifacts/polyvore_core7_v2/core7_drop_v2/core7_item_metadata_v1_{train,valid,test}.jsonl \
  --output artifacts/retrieval_catalog.jsonl
```

It must verify all 78,182 Core-7 images; an incomplete or fabricated catalog is not a production fallback. Make artifacts readable by UID 10001 (directories executable/readable, files readable). Set `ARTIFACT_HOST_DIR` to an absolute path if provisioned elsewhere.

## Launch

```bash
docker compose -f docker-compose.gpu.yml config --quiet
docker compose -f docker-compose.gpu.yml up -d --build
curl --fail http://localhost:57260/api/health
curl --fail http://localhost:57260/api/ready
```

Readiness returns 503 until the catalog, FashionCLIP and Qwen are fully loaded. First startup may download weights and take several minutes. Watch `docker compose -f docker-compose.gpu.yml logs -f backend`; safe error types distinguish initialization failure from warmup. The named HF cache persists across restarts. Compose mounts artifacts read-only and does not publish backend port 8000.

Website after successful deployment: `http://SERVER_IP:57260`. Replace SERVER_IP with the actual host; this is not a verified live URL. Port 57260 must be available. If the old project owns it, stop that project's frontend before starting this one; do not run both on the same port. For public HTTPS, put your existing TLS reverse proxy in front of this port. There is no authentication in this MVP.

## Mandatory GPU smoke

Use a consented photo with one person, a plain background and at least three visible canonical items:

```bash
python -m pip install httpx==0.28.1
python scripts/gpu_smoke.py --url http://localhost:57260 --image /absolute/path/outfit.jpg --expect ok
python scripts/gpu_smoke.py --url http://localhost:57260 --image /absolute/path/two_people.jpg --expect rejected
python scripts/gpu_smoke.py --url http://localhost:57260 --image /absolute/path/dress_and_shoes.jpg --expect rejected
```

The script checks readiness, successful API contract, exact score sum, three unique same-category items, and downloads/decodes each real image. Inspect the returned explanations against actual images and separately test mannequin/background cases. Record latency and peak VRAM with `nvidia-smi`. A single smoke pass does not establish perceptual accuracy; maintain a small labeled scene/outfit acceptance set before wider rollout.

## Local development without Docker

Install CUDA packages on the GPU machine instead of CPU PyTorch:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
set -a; source .env; set +a
PYTHONPATH=backend python -m uvicorn app.main:app --host 127.0.0.1 --port "${APP_PORT:-8000}" --workers 1
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Vite proxies `/api` to `127.0.0.1:8000`. If APP_PORT changes, update the development proxy accordingly. Production backend port is deliberately fixed at 8000; public port is 57260. Keep a single worker; multiple workers reload the large model and defeat process-local concurrency control.

## Troubleshooting

| Symptom | Action |
|---|---|
| Readiness stays 503 | Check logs, mount paths, cache SHA, catalog image files, network/model access and CUDA. Restart after fixing startup failures. |
| HF cache permission denied | Named volume must be writable by UID 10001. New images initialize it with that ownership; correct existing volume ownership if reused. |
| GPU out of memory | Keep concurrency 1, lower VISION_MAX_PIXELS or MAX_NEW_TOKENS, use FP16 if needed, and restart. Never add Uvicorn workers. |
| Too few/missing candidates | Re-run preparation with all three metadata files and complete images; do not use placeholders. |
| VLM_OUTPUT_ERROR | Try a clearer photo, inspect prompt/contract version, then evaluate repeated failures on the GPU. Raw model output is intentionally hidden. |
| Browser cannot reach backend | Use the frontend URL and relative `/api`; do not configure browser localhost:8000. |
| Timeout then busy | A timed-out GPU job retains its slot until it stops. Retry later; inspect GPU health if it does not clear. |

Rolling back: check out a known commit, keep its compatible prepared artifacts, rebuild Compose. `docker compose down` stops services; do not add `-v` unless you intentionally want to delete cached model weights.
