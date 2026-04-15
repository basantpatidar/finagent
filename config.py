from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    claude_model: str = Field("claude-3-sonnet-20240229", env="CLAUDE_MODEL")

    # Agent behaviour
    agent_max_iterations: int = Field(10, env="AGENT_MAX_ITERATIONS")
    agent_temperature: float = Field(0.1, env="AGENT_TEMPERATURE")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_debug: bool = Field(False, env="API_DEBUG")

    # Financial data
    yfinance_timeout: int = Field(10, env="YFINANCE_TIMEOUT")
    news_max_articles: int = Field(5, env="NEWS_MAX_ARTICLES")
    sec_max_filings: int = Field(3, env="SEC_MAX_FILINGS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
