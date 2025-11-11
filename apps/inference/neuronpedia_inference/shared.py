import asyncio
from functools import wraps

import torch

# from transformer_lens.model_bridge import TransformerBridge
from nnterp import StandardizedTransformer
from transformer_lens import HookedTransformer

request_lock = asyncio.Lock()


def with_request_lock():
    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            async with request_lock:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class Model:
    _instance: (
        HookedTransformer | StandardizedTransformer
    )  # | TransformerBridge  # type: ignore

    @classmethod
    def get_instance(
        cls,
    ) -> HookedTransformer | StandardizedTransformer:  # | TransformerBridge:
        if cls._instance is None:
            raise ValueError("Model not initialized")
        return cls._instance

    @classmethod
    def set_instance(
        cls,
        model: HookedTransformer | StandardizedTransformer,  # | TransformerBridge
    ) -> None:
        cls._instance = model


MODEL = Model()

STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


TLENS_MODEL_ID_TO_HF_MODEL_ID = {
    "gpt2-small": "openai-community/gpt2",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
}


def replace_tlens_model_id_with_hf_model_id(model_id: str) -> str:
    if model_id in TLENS_MODEL_ID_TO_HF_MODEL_ID:
        return TLENS_MODEL_ID_TO_HF_MODEL_ID[model_id]
    return model_id
