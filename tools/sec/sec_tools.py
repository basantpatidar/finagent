import logging
from typing import Any

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
HEADERS = {
    "User-Agent": "FinAgent research-tool contact@finagent.dev",
    "Accept-Encoding": "gzip, deflate",
}


def _get_cik_for_ticker(ticker: str) -> str | None:
    """Map ticker symbol to SEC CIK number via EDGAR company search."""
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker.upper()}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
        resp = requests.get(
            f"{EDGAR_BASE}/files/company_tickers.json",
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                logger.info("[SECTool] Resolved ticker=%s to CIK=%s", ticker, cik)
                return cik

        return None
    except Exception as e:
        logger.warning("[SECTool] CIK lookup failed for %s: %s", ticker, e)
        return None


@tool
def get_sec_filings(ticker: str, max_filings: int = 3) -> dict[str, Any]:
    """
    Fetch recent SEC filings (10-K annual reports and 10-Q quarterly reports)
    for a company via the SEC EDGAR API.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'
        max_filings: Maximum number of filings to return (default 3)

    Returns:
        Dict with list of recent filings including type, date, and EDGAR URL.
    """
    try:
        logger.info("[SECTool] Fetching SEC filings for ticker=%s", ticker)

        cik = _get_cik_for_ticker(ticker)
        if not cik:
            return {
                "error": f"Could not resolve CIK for ticker '{ticker}'",
                "ticker": ticker,
                "filings": [],
            }

        # Fetch submission history from EDGAR
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])

        target_forms = {"10-K", "10-Q"}
        filings = []

        for form, date, accession, doc in zip(forms, dates, accessions, descriptions):
            if form in target_forms and len(filings) < max_filings:
                accession_clean = accession.replace("-", "")
                edgar_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession_clean}/{doc}"
                )
                filings.append({
                    "form_type": form,
                    "filing_date": date,
                    "accession_number": accession,
                    "document_url": edgar_url,
                    "description": f"{form} filed on {date}",
                })

        company_name = data.get("name", ticker.upper())

        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "cik": cik,
            "filing_count": len(filings),
            "filings": filings,
        }

    except Exception as e:
        logger.error("[SECTool] Error fetching filings for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker, "filings": []}


@tool
def get_sec_facts(ticker: str) -> dict[str, Any]:
    """
    Fetch key financial facts from SEC XBRL data: revenue, net income,
    total assets, and operating cash flow from the most recent filings.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with key financial metrics extracted from SEC XBRL data.
    """
    try:
        logger.info("[SECTool] Fetching XBRL facts for ticker=%s", ticker)

        cik = _get_cik_for_ticker(ticker)
        if not cik:
            return {"error": f"Could not resolve CIK for '{ticker}'", "ticker": ticker}

        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        facts = data.get("facts", {}).get("us-gaap", {})

        def _latest_annual(concept: str) -> dict | None:
            """Extract the most recent annual (10-K) value for a concept."""
            entries = facts.get(concept, {}).get("units", {})
            usd_entries = entries.get("USD", [])
            annual = [
                e for e in usd_entries
                if e.get("form") == "10-K" and e.get("val") is not None
            ]
            if not annual:
                return None
            latest = sorted(annual, key=lambda x: x.get("end", ""), reverse=True)[0]
            return {
                "value": latest.get("val"),
                "period_end": latest.get("end"),
                "filed": latest.get("filed"),
            }

        return {
            "ticker": ticker.upper(),
            "revenue": _latest_annual("Revenues") or _latest_annual("RevenueFromContractWithCustomerExcludingAssessedTax"),
            "net_income": _latest_annual("NetIncomeLoss"),
            "total_assets": _latest_annual("Assets"),
            "total_liabilities": _latest_annual("Liabilities"),
            "operating_cash_flow": _latest_annual("NetCashProvidedByUsedInOperatingActivities"),
            "research_and_development": _latest_annual("ResearchAndDevelopmentExpense"),
        }

    except Exception as e:
        logger.error("[SECTool] Error fetching XBRL facts for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker}
