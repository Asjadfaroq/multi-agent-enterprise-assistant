from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    anthropic_api_key: SecretStr = SecretStr("")
    database_url: str = "postgresql://assistant:assistant@localhost:5433/assistant"

    # Set per process by whoever launches the knowledge server; the model never supplies it.
    assistant_user_id: str = ""

    agent_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-sonnet-5-5"
    max_review_revisions: int = Field(default=1, ge=0)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    # bge models are trained with this instruction on queries; documents are embedded without it.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    search_result_count: int = Field(default=5, gt=0)

    # bge-small truncates input at 512 tokens, so chunks must stay below that.
    chunk_size_tokens: int = Field(default=500, gt=0, le=510)

    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
