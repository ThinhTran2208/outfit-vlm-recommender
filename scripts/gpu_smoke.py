#!/usr/bin/env python3
"""Exercise the deployed real pipeline and each actual recommendation image."""

import argparse
import io
import mimetypes
import sys
from pathlib import Path
import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.schemas import Success, Rejected

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:57260")
p.add_argument("--image", required=True)
p.add_argument("--expect", choices=["ok", "rejected"], default="ok")
a = p.parse_args()
with httpx.Client(base_url=a.url, timeout=240) as client:
    client.get("/api/ready").raise_for_status()
    with open(a.image, "rb") as f:
        r = client.post(
            "/api/analyze", files={"image": (Path(a.image).name, f, mimetypes.guess_type(a.image)[0])}
        )
    r.raise_for_status()
    data = r.json()
    assert data["status"] == a.expect, data
    if a.expect == "ok":
        result = Success.model_validate(data)
        assert result.score.total == result.score.dimensions.total
        assert len({x.item_id for x in result.recommendations}) == 3
        for item in result.recommendations:
            assert item.category == result.problematic_item.category
            response = client.get(item.image_url)
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as im:
                im.verify()
        print("PASS: real API, rubric, three unique same-category items and decoded images")
    else:
        Rejected.model_validate(data)
        assert "score" not in data and "recommendations" not in data
        print("PASS: rejected without score/recommendations", data["error_code"])
