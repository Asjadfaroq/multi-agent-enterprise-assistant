from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from assistant.config import get_settings

Embeddings = NDArray[np.float32]


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


def embed_texts(texts: list[str]) -> Embeddings:
    return get_embedding_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def count_tokens(text: str) -> int:
    return len(get_embedding_model().tokenizer.encode(text, add_special_tokens=False))
