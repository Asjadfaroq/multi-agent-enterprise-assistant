from functools import lru_cache
from typing import Literal

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from assistant.config import get_settings

FloatArray = NDArray[np.float32]
ModelRole = Literal["agent", "judge"]


def get_chat_model(role: ModelRole) -> BaseChatModel:
    settings = get_settings()
    if not settings.anthropic_api_key.get_secret_value():
        raise ValueError("ANTHROPIC_API_KEY is not set; add it to .env")
    model_by_role = {"agent": settings.agent_model, "judge": settings.judge_model}
    return ChatAnthropic(model=model_by_role[role], api_key=settings.anthropic_api_key)


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    actual_dimensions = model.get_embedding_dimension()
    if actual_dimensions != settings.embedding_dimensions:
        raise ValueError(
            f"Embedding model {settings.embedding_model} produces {actual_dimensions}-dim vectors "
            f"but EMBEDDING_DIMENSIONS is {settings.embedding_dimensions}"
        )
    return model


def embed_texts(texts: list[str]) -> FloatArray:
    return get_embedding_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def embed_query(query: str) -> FloatArray:
    return embed_texts([get_settings().embedding_query_prefix + query])[0]


def count_tokens(text: str) -> int:
    return len(get_embedding_model().tokenizer.encode(text, add_special_tokens=False))
