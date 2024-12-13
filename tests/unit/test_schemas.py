import pytest
from pydantic import ValidationError


class TestResearchRequestValidation:

    def test_valid_ticker_accepted(self):
        from finagent.api.models.schemas import ResearchRequest
        req = ResearchRequest(ticker="AAPL")
        assert req.ticker == "AAPL"

    def test_ticker_lowercased_to_upper(self):
        from finagent.api.models.schemas import ResearchRequest
        req = ResearchRequest(ticker="aapl")
        assert req.ticker == "AAPL"

    def test_ticker_with_whitespace_stripped(self):
        from finagent.api.models.schemas import ResearchRequest
        req = ResearchRequest(ticker=" MSFT ")
        assert req.ticker == "MSFT"

    def test_numeric_ticker_rejected(self):
        from finagent.api.models.schemas import ResearchRequest
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="123")

    def test_ticker_with_special_chars_rejected(self):
        from finagent.api.models.schemas import ResearchRequest
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="AA.PL")

    def test_empty_ticker_rejected(self):
        from finagent.api.models.schemas import ResearchRequest
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="")

    def test_ticker_too_long_rejected(self):
        from finagent.api.models.schemas import ResearchRequest
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="TOOLONGTICKER")

    def test_optional_question_defaults_to_none(self):
        from finagent.api.models.schemas import ResearchRequest
        req = ResearchRequest(ticker="AAPL")
        assert req.question is None

    def test_question_accepted_when_provided(self):
        from finagent.api.models.schemas import ResearchRequest
        req = ResearchRequest(ticker="AAPL", question="What is the P/E ratio?")
        assert req.question == "What is the P/E ratio?"

    def test_question_too_long_rejected(self):
        from finagent.api.models.schemas import ResearchRequest
        with pytest.raises(ValidationError):
            ResearchRequest(ticker="AAPL", question="x" * 501)
