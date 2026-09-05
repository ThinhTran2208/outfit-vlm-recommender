import asyncio
import io
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.exceptions import HTTPException
from app.config import Settings
from app.errors import AppError
from app.schemas import Success, Rejected, Error
from app.services.pipeline import stage


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "level": record.levelname,
                "event": record.getMessage(),
                "request_id": getattr(record, "request_id", None),
                "timings": getattr(record, "timings", None),
                "error_type": getattr(record, "error_type", None),
            }
        )


log = logging.getLogger("outfit")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    log.addHandler(h)
    log.setLevel(logging.INFO)


class UploadLimit:
    """Bound complete multipart body before parser spools data to disk."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            try:
                message = await asyncio.wait_for(receive(), timeout=30)
            except TimeoutError:
                return await JSONResponse(
                    Error(error_code="REQUEST_TIMEOUT", message_vi="Tải ảnh quá thời gian.").model_dump(),
                    status_code=408,
                )(scope, receive, send)
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                return await JSONResponse(
                    Error(
                        error_code="INVALID_IMAGE", message_vi="Ảnh vượt quá dung lượng cho phép."
                    ).model_dump(),
                    status_code=413,
                )(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        i = 0

        async def replay():
            nonlocal i
            if i < len(chunks):
                msg = chunks[i]
                i += 1
                return msg
            return await receive()

        await self.app(scope, replay, send)


def decode_image(data, filename, mime, settings):
    formats = {
        ".jpg": ("image/jpeg", "JPEG"),
        ".jpeg": ("image/jpeg", "JPEG"),
        ".png": ("image/png", "PNG"),
        ".webp": ("image/webp", "WEBP"),
    }
    expected = formats.get(Path(filename or "").suffix.lower())
    if not expected or mime != expected[0] or not data or len(data) > settings.max_upload_mb * 1024 * 1024:
        raise AppError("INVALID_IMAGE", "Vui lòng chọn ảnh JPG, PNG hoặc WebP hợp lệ.", 400)
    try:
        with Image.open(io.BytesIO(data)) as im:
            if (
                im.format != expected[1]
                or im.width * im.height > settings.max_image_pixels
                or getattr(im, "n_frames", 1) != 1
            ):
                raise ValueError("Invalid format, dimensions, or animated image")
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise AppError("INVALID_IMAGE", "Không thể đọc ảnh. Vui lòng chọn ảnh khác.", 400) from exc


def create_app(settings=None, pipeline=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = pipeline
        app.state.pool = ThreadPoolExecutor(max_workers=settings.max_concurrent_vlm_requests)
        app.state.slots = asyncio.Semaphore(settings.max_concurrent_vlm_requests)

        async def startup():
            if pipeline is not None:
                return

            def load():
                from app.services.catalog import Catalog
                from app.services.models import QwenBackend, StructuredVLM, FashionCLIP
                from app.services.pipeline import Pipeline

                catalog, manifest = Catalog.load(settings)
                encoder = FashionCLIP(manifest, settings.fashionclip_revision)
                vlm = StructuredVLM(QwenBackend(settings))
                return Pipeline(vlm, encoder, catalog, settings)

            try:
                app.state.pipeline = await asyncio.get_running_loop().run_in_executor(app.state.pool, load)
            except Exception as exc:
                log.error("startup_failed", extra={"error_type": type(exc).__name__})

        task = asyncio.create_task(startup())
        yield
        await task
        app.state.pipeline = None
        app.state.pool.shutdown(wait=True)

    app = FastAPI(title="Outfit Advisor", version="1.0.0", lifespan=lifespan)
    app.add_middleware(UploadLimit, limit=(settings.max_upload_mb * 1024 * 1024) + 65536)

    @app.middleware("http")
    async def headers(request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        rid = getattr(request.state, "request_id", None)
        log.warning(exc.code, extra={"request_id": rid})
        return JSONResponse(
            Error(error_code=exc.code, message_vi=exc.message, request_id=rid).model_dump(),
            status_code=exc.status,
        )

    @app.exception_handler(Exception)
    async def unknown_error(request, exc):
        log.error("internal_error", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            Error(
                error_code="INTERNAL_ERROR",
                message_vi="Hệ thống gặp lỗi. Vui lòng thử lại.",
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
            status_code=500,
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/ready")
    def ready(request: Request):
        if request.app.state.pipeline is None:
            raise AppError("MODEL_NOT_READY", "Mô hình đang khởi động hoặc dữ liệu chưa sẵn sàng.")
        return {"status": "ready"}

    @app.post(
        "/api/analyze",
        response_model=Success | Rejected,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["image"],
                            "properties": {"image": {"type": "string", "format": "binary"}},
                        }
                    }
                },
            }
        },
        responses={
            400: {"model": Error},
            502: {"model": Error},
            503: {"model": Error},
            504: {"model": Error},
        },
    )
    async def analyze(request: Request):
        p = request.app.state.pipeline
        if p is None:
            raise AppError("MODEL_NOT_READY", "Mô hình đang khởi động hoặc dữ liệu chưa sẵn sàng.")
        try:
            await asyncio.wait_for(app.state.slots.acquire(), timeout=settings.queue_timeout_seconds)
        except TimeoutError as exc:
            raise AppError("SERVER_BUSY", "Hệ thống đang xử lý ảnh khác. Vui lòng thử lại sau.", 429) from exc
        submitted = False
        image = None
        try:
            timings = {}
            with stage(timings, "image_validation_ms"):
                try:
                    async with request.form(
                        max_files=1, max_fields=0, max_part_size=settings.max_upload_mb * 1024 * 1024
                    ) as form:
                        upload = form.get("image")
                        if upload is None or not hasattr(upload, "read") or len(form) != 1:
                            raise AppError("INVALID_IMAGE", "Vui lòng tải một ảnh outfit.", 400)
                        data = await upload.read(settings.max_upload_mb * 1024 * 1024 + 1)
                        image = decode_image(data, upload.filename, upload.content_type, settings)
                except HTTPException as exc:
                    raise AppError("INVALID_IMAGE", "Dữ liệu ảnh tải lên không hợp lệ.", 400) from exc
            deadline = time.monotonic() + settings.request_timeout_seconds

            def run():
                try:
                    return p.analyze(image, request.state.request_id, deadline, timings)
                finally:
                    image.close()

            future = asyncio.get_running_loop().run_in_executor(app.state.pool, run)
            submitted = True

            def finished(f):
                app.state.slots.release()
                if not f.cancelled():
                    f.exception()  # consume exceptions even after client timeout/disconnect

            future.add_done_callback(finished)
            try:
                return await asyncio.wait_for(asyncio.shield(future), settings.request_timeout_seconds)
            except TimeoutError as exc:
                raise AppError("REQUEST_TIMEOUT", "Phân tích quá thời gian. Vui lòng thử lại.", 504) from exc
        finally:
            if not submitted:
                if image is not None:
                    image.close()
                app.state.slots.release()

    @app.get("/api/items/{item_id}/image")
    def item_image(item_id: str, request: Request):
        p = request.app.state.pipeline
        if p is None:
            raise AppError("MODEL_NOT_READY", "Dữ liệu chưa sẵn sàng.")
        if item_id not in p.catalog.records:
            raise AppError("RETRIEVAL_ERROR", "Không tìm thấy ảnh sản phẩm.", 404)
        try:
            with p.catalog.image(item_id) as im:
                im.thumbnail((800, 800))
                out = io.BytesIO()
                im.save(out, format="JPEG", quality=90)
            return Response(out.getvalue(), media_type="image/jpeg")
        except (ValueError, OSError, KeyError) as exc:
            raise AppError("RETRIEVAL_ERROR", "Không thể đọc ảnh sản phẩm.", 503) from exc

    return app


app = create_app()
