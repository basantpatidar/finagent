from pydantic import BaseModel, Field, field_validator
import re


class ResearchRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol e.g. 'AAPL'", min_length=1, max_length=10)
    question: str | None = Field(
        None,
        description="Optional custom research question. Defaults to full research brief.",
        max_length=500,
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r"^[A-Z]{1,10}$", v):
            raise ValueError("Ticker must be 1-10 uppercase letters only (e.g. AAPL, MSFT)")
        return v


class ResearchResponse(BaseModel):
    ticker: str
    research_brief: str
    errors: list[str] = Field(default_factory=list)
    tool_calls_made: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    version: str = "1.0.0"


class StreamEvent(BaseModel):
    type: str  # tool_start | tool_end | text_chunk | complete | error
    data: dict
