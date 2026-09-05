import json
import re
import time
from pathlib import Path
from pydantic import ValidationError
from app.errors import AppError

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def parse_json(raw):
    text = raw.strip()
    # Safe framing repair only; no eval, no invented fields or scores.
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    decoder = json.JSONDecoder()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        obj, end = decoder.raw_decode(text[start:])
        if "{" in text[start + end :] or "}" in text[start + end :]:
            raise ValueError("Multiple or malformed JSON objects")
        return obj


class StructuredVLM:
    def __init__(self, backend):
        self.backend = backend

    def call(self, prompt_name, images, context, schema, deadline, check=None):
        system = (PROMPTS / prompt_name).read_text()
        system += "\nRequired JSON schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        correction = ""
        for attempt in range(2):
            if time.monotonic() >= deadline:
                raise AppError("REQUEST_TIMEOUT", "Phân tích quá thời gian. Vui lòng thử lại.", 504)
            raw = self.backend.generate(system + correction, images, context, deadline)
            try:
                result = schema.model_validate(parse_json(raw))
                if check:
                    check(result)
                return result
            except (ValidationError, ValueError, TypeError, KeyError):
                correction = "\nRETRY: previous response violated the contract. Re-evaluate the images and return exactly one valid JSON object. Check counts, total, categories, mode, IDs, and all required fields. No prose or Markdown."
        raise AppError("VLM_OUTPUT_ERROR", "Hệ thống chưa tạo được kết quả hợp lệ. Vui lòng thử lại.", 502)


class QwenBackend:
    def __init__(self, settings):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen requires NVIDIA CUDA; CPU mocks are test-only")
        self.torch = torch
        self.settings = settings
        dtype = settings.qwen_dtype
        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            settings.qwen_model_id,
            revision=settings.qwen_revision,
            use_safetensors=True,
            dtype=getattr(torch, dtype),
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            settings.qwen_model_id, revision=settings.qwen_revision
        )

    def generate(self, system, images, context, deadline):
        from qwen_vl_utils import process_vision_info
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        content = []
        for label, im in images:
            content.extend(
                [
                    {"type": "text", "text": label},
                    {
                        "type": "image",
                        "image": im,
                        "min_pixels": 50176,
                        "max_pixels": self.settings.vision_max_pixels,
                    },
                ]
            )
        content.append({"type": "text", "text": json.dumps(context, ensure_ascii=False)})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=prompt, images=image_inputs, videos=video_inputs, do_resize=False, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.settings.max_new_tokens,
                stopping_criteria=StoppingCriteriaList([Deadline()]),
            )
        if time.monotonic() >= deadline:
            raise AppError("REQUEST_TIMEOUT", "Phân tích quá thời gian. Vui lòng thử lại.", 504)
        return self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class FashionCLIP:
    def __init__(self, manifest, revision):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.dimension = manifest["embedding_dimension"]
        # CPU text encoding avoids competing with Qwen for GPU VRAM.
        self.model = (
            CLIPModel.from_pretrained(manifest["model_name_or_version"], revision=revision, weights_only=True)
            .eval()
            .to("cpu")
        )
        self.processor = CLIPProcessor.from_pretrained(manifest["model_name_or_version"], revision=revision)
        if self.model.config.projection_dim != self.dimension:
            raise ValueError("FashionCLIP projection dimension differs from manifest")

    def encode(self, queries):
        inputs = self.processor(text=queries, padding=True, truncation=True, return_tensors="pt")
        with self.torch.inference_mode():
            vectors = self.model.get_text_features(**inputs)
            vectors = getattr(vectors, "pooler_output", vectors).float()
            vectors = self.torch.nn.functional.normalize(vectors, dim=-1)
        if vectors.shape != (3, self.dimension) or not self.torch.isfinite(vectors).all():
            raise ValueError("Invalid FashionCLIP features")
        return vectors.numpy()
