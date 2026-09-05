from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    qwen_model_id: Literal["Qwen/Qwen3-VL-4B-Instruct"] = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_revision: str = "main"
    qwen_dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    fashionclip_revision: str = "main"
    artifact_dir: Path = Path("artifacts")
    fashionclip_embeddings_path: Path = Path("artifacts/fashionclip_item_embeddings.pt")
    embedding_manifest_path: Path = Path("artifacts/embedding_manifest_v1.json")
    retrieval_catalog_path: Path = Path("artifacts/retrieval_catalog.jsonl")
    max_concurrent_vlm_requests: int = Field(1, ge=1, le=4)
    high_score_similarity_threshold: int = Field(85, ge=0, le=100)
    max_upload_mb: int = Field(10, ge=1, le=30)
    max_image_pixels: int = Field(20000000, ge=1)
    max_new_tokens: int = Field(2500, ge=256, le=6000)
    vision_max_pixels: int = Field(802816, ge=50176)
    request_timeout_seconds: int = Field(180, ge=1)
    queue_timeout_seconds: int = Field(2, ge=1)
    app_port: int = 8000
