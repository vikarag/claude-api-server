from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    api_secret_token: str = "changeme-generate-a-real-token"
    claude_cli_path: str = "claude"
    default_model: str = "sonnet"
    max_concurrent: int = 2
    request_timeout: int = 120
    allowed_paths: str = "."
    max_budget_usd: float = 1.0

    @property
    def allowed_path_list(self) -> list[Path]:
        return [Path(p.strip()) for p in self.allowed_paths.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
