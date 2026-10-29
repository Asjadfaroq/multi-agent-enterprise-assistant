from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    anthropic_api_key: SecretStr = SecretStr("")
    database_url: str = "postgresql://assistant:assistant@localhost:5433/assistant"

    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8001
    # Shared secret the API presents to the MCP server, so X-User is only trusted from the API.
    mcp_service_token: SecretStr = SecretStr("")
    api_url: str = "http://127.0.0.1:8000"

    agent_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-sonnet-5-5"
    max_review_revisions: int = Field(default=1, ge=0)
    max_profile_facts_per_turn: int = Field(default=2, gt=0)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    # bge models are trained with this instruction on queries; documents are embedded without it.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    search_result_count: int = Field(default=5, gt=0)
    ticket_list_limit: int = Field(default=20, gt=0)

    # Base64-looking runs at least this long are decoded and checked for hidden instructions.
    injection_encoded_payload_min_length: int = Field(default=40, gt=0)
    pii_phone_regions: tuple[str, ...] = ("SA", "PT", "US", "GB")

    # bge-small truncates input at 512 tokens, so chunks must stay below that.
    chunk_size_tokens: int = Field(default=500, gt=0, le=510)

    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"

    # Gates for evals/regression_test.py; a case also fails if any judge score is below the floor.
    eval_min_faithfulness: float = Field(default=0.85, ge=0, le=1)
    eval_min_context_precision: float = Field(default=0.80, ge=0, le=1)
    eval_min_judge_mean: float = Field(default=4.0, ge=1, le=5)
    eval_min_behaviour_pass_rate: float = Field(default=1.0, ge=0, le=1)
    eval_case_min_judge_score: int = Field(default=3, ge=1, le=5)
    eval_arabic_min_letter_ratio: float = Field(default=0.5, ge=0, le=1)
    eval_scoring_concurrency: int = Field(default=4, gt=0)

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    @property
    def mcp_server_url(self) -> str:
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}/mcp"


@lru_cache
def get_settings() -> Settings:
    return Settings()
