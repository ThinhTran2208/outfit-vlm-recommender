import numpy as np
import torch
from transformers import CLIPConfig, CLIPModel
from app.services.models import FashionCLIP


def test_projected_text_embedding_adapter_without_download():
    # Real tiny Transformers model checks the installed feature-return API,
    # projection shape, batch handling and normalization without HF weights.
    config = CLIPConfig(
        text_config={
            "vocab_size": 20,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 10,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 8,
        },
        projection_dim=8,
    )
    adapter = FashionCLIP.__new__(FashionCLIP)
    adapter.torch = torch
    adapter.dimension = 8
    adapter.model = CLIPModel(config).eval()
    calls = []

    def processor(**kwargs):
        calls.append(kwargs)
        return {
            "input_ids": torch.tensor([[1, 3, 2], [1, 4, 2], [1, 5, 2]]),
            "attention_mask": torch.ones((3, 3), dtype=torch.int64),
        }

    adapter.processor = processor
    output = adapter.encode(["black shoes", "white shoes", "brown shoes"])
    assert output.shape == (3, 8) and np.isfinite(output).all()
    assert np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-6)
    assert len(calls) == 1 and len(calls[0]["text"]) == 3
