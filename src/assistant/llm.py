from functools import lru_cache
from typing import Literal

import numpy as np
from huggingface_hub import try_to_load_from_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from numpy.typing import NDArray
from sentence_transformers import CrossEncoder, SentenceTransformer

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
    model = SentenceTransformer(
        settings.embedding_model,
        device="cpu",
        local_files_only=is_cached_locally(settings.embedding_model),
    )
    actual_dimensions = model.get_embedding_dimension()
    if actual_dimensions != settings.embedding_dimensions:
        raise ValueError(
            f"Embedding model {settings.embedding_model} produces {actual_dimensions}-dim vectors "
            f"but EMBEDDING_DIMENSIONS is {settings.embedding_dimensions}"
        )
    return model


def is_cached_locally(model_name: str) -> bool:
    # A cached model is loaded without contacting the Hugging Face Hub, which adds seconds.
    return isinstance(try_to_load_from_cache(model_name, "config.json"), str)


@lru_cache
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(
        settings.reranker_model,
        device="cpu",
        max_length=settings.reranker_max_tokens,
        local_files_only=is_cached_locally(settings.reranker_model),
    )


def embed_texts(texts: list[str]) -> FloatArray:
    return get_embedding_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )


def embed_query(query: str) -> FloatArray:
    return embed_texts([get_settings().embedding_query_prefix + query])[0]


def embed_passages(passages: list[str]) -> FloatArray:
    prefix = get_settings().embedding_passage_prefix
    return embed_texts([prefix + passage for passage in passages])


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    return get_reranker().predict([(query, passage) for passage in passages]).tolist()


def count_tokens(text: str) -> int:
    return len(get_embedding_model().tokenizer.encode(text, add_special_tokens=False))
