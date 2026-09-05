import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from app.main import create_app
from app.config import Settings


def test_api_success(analysis, make_pipeline, jpeg):
    p, _, _ = make_pipeline(analysis)
    with TestClient(create_app(pipeline=p)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/ready").status_code == 200
        r = client.post("/api/analyze", files={"image": ("a.jpg", jpeg, "image/jpeg")})
        assert r.status_code == 200, r.text
        assert r.json()["score"]["total"] == 80
        assert "x-request-id" in r.headers
        for item in r.json()["recommendations"]:
            image = client.get(item["image_url"])
            assert image.status_code == 200
            Image.open(io.BytesIO(image.content)).verify()
        assert client.get("/api/items/unknown/image").status_code == 404
        assert client.get("/api/health").headers.get("access-control-allow-origin") is None


@pytest.mark.parametrize(
    "name,mime,data",
    [
        ("a.txt", "image/jpeg", b"bad"),
        ("a.jpg", "text/plain", b"bad"),
        ("a.jpg", "image/jpeg", b"bad"),
        ("a.png", "image/png", None),
    ],
)
def test_invalid_upload(analysis, make_pipeline, jpeg, name, mime, data):
    p, b, _ = make_pipeline(analysis)
    with TestClient(create_app(pipeline=p)) as client:
        r = client.post("/api/analyze", files={"image": (name, data or jpeg, mime)})
        assert r.status_code == 400 and r.json()["error_code"] == "INVALID_IMAGE"
        assert not b.calls


def test_upload_limit(analysis, make_pipeline):
    p, b, _ = make_pipeline(analysis)
    with TestClient(create_app(Settings(max_upload_mb=1), p)) as client:
        r = client.post(
            "/api/analyze", files={"image": ("a.jpg", b"x" * (1024 * 1024 + 70000), "image/jpeg")}
        )
        assert r.status_code == 413 and not b.calls


def test_missing_upload(analysis, make_pipeline):
    p, b, _ = make_pipeline(analysis)
    with TestClient(create_app(pipeline=p)) as client:
        r = client.post("/api/analyze")
        assert r.status_code == 400 and not b.calls


def test_not_ready(analysis, make_pipeline):
    p, _, _ = make_pipeline(analysis)
    with TestClient(create_app(pipeline=p)) as client:
        client.app.state.pipeline = None
        assert client.get("/api/ready").status_code == 503
        assert client.post("/api/analyze").json()["error_code"] == "MODEL_NOT_READY"


def test_timeout_retains_slot_until_worker_finishes(analysis, make_pipeline, jpeg):
    import threading

    p, _, _ = make_pipeline(analysis)
    started = threading.Event()
    release = threading.Event()

    def blocked(*args):
        started.set()
        release.wait(timeout=5)
        return {
            "status": "rejected",
            "error_code": "NOT_ENOUGH_ITEMS",
            "message_vi": "Không đủ item để đánh giá.",
            "counted_item_count": 0,
            "garments": [],
        }

    p.analyze = blocked
    with TestClient(create_app(Settings(request_timeout_seconds=1, queue_timeout_seconds=1), p)) as client:
        try:
            r = client.post("/api/analyze", files={"image": ("a.jpg", jpeg, "image/jpeg")})
            assert r.status_code == 504 and started.is_set()
            assert client.app.state.slots.locked()
            r = client.post("/api/analyze", files={"image": ("a.jpg", jpeg, "image/jpeg")})
            assert r.status_code == 429
        finally:
            release.set()
