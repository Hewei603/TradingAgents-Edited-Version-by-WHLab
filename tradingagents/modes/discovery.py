"""Prompt-driven stock discovery mode.

The mode intentionally stays outside the graph: it creates a candidate list,
then uses the existing single-ticker TradingAgentsGraph for deep analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from tradingagents.llm_clients import create_llm_client
from tradingagents.modes.common import ModeRunConfig, run_single_ticker


_TICKER_RE = re.compile(r"\b[A-Z][A-Z0-9.\-]{0,9}\b")
_COMMON_WORDS = {
    "AI", "API", "CPI", "ETF", "EV", "FCF", "GDP", "IPO", "LLM", "M&A",
    "NASDAQ", "NYSE", "OTC", "SEC", "US", "USA", "USD",
}


@dataclass
class DiscoveryRequest:
    prompt: str
    stock_count: int = 5
    market: str = "US"
    risk_level: str = "aggressive"
    timeframe: str = "3-12 months"
    candidate_pool_size: int = 12


def _extract_json_block(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return json.loads(match.group(1))
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Could not parse candidate JSON")


def _normalise_candidates(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        raw = raw.get("candidates", [])
    candidates: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return candidates
    for item in raw:
        if isinstance(item, str):
            ticker = item.upper().strip()
            candidates.append({"ticker": ticker, "reason": "LLM candidate"})
            continue
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker or ticker in _COMMON_WORDS:
            continue
        candidates.append({
            "ticker": ticker,
            "company": str(item.get("company", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
            "catalyst": str(item.get("catalyst", "")).strip(),
            "risk": str(item.get("risk", "")).strip(),
        })
    deduped = []
    seen = set()
    for candidate in candidates:
        ticker = candidate["ticker"]
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(candidate)
    return deduped


def _fallback_tickers(text: str) -> list[dict[str, str]]:
    tickers = []
    seen = set()
    for match in _TICKER_RE.findall(text.upper()):
        if match in _COMMON_WORDS or match in seen:
            continue
        seen.add(match)
        tickers.append({"ticker": match, "reason": "Extracted from model text"})
    return tickers


def generate_candidates(request: DiscoveryRequest, run_config: ModeRunConfig) -> list[dict[str, str]]:
    llm_kwargs: dict[str, Any] = {}
    if run_config.llm_provider == "google" and run_config.google_thinking_level:
        llm_kwargs["thinking_level"] = run_config.google_thinking_level
    if run_config.llm_provider == "openai" and run_config.openai_reasoning_effort:
        llm_kwargs["reasoning_effort"] = run_config.openai_reasoning_effort
    if run_config.llm_provider == "anthropic" and run_config.anthropic_effort:
        llm_kwargs["effort"] = run_config.anthropic_effort

    llm = create_llm_client(
        provider=run_config.llm_provider,
        model=run_config.quick_model or "gpt-5.4-mini",
        base_url=run_config.backend_url,
        **llm_kwargs,
    )
    prompt = f"""
You are a stock discovery analyst. Generate a grounded candidate list for a
TradingAgents deep-dive workflow. The user did not provide tickers.

User request:
{request.prompt}

Constraints:
- Market: {request.market}
- Risk level: {request.risk_level}
- Desired timeframe: {request.timeframe}
- Return {request.candidate_pool_size} candidates, biased toward liquid listed equities.
- Prefer candidates with plausible 3-12 month re-rating or 2x paths when the prompt asks for aggressive upside.
- Do not claim certainty. Do not invent private information.
- Avoid megacaps unless the user explicitly asks for them or the setup is unusually asymmetric.

Return only JSON, as an array of objects with:
ticker, company, reason, catalyst, risk.
"""
    response = llm.invoke(prompt)
    text = getattr(response, "content", str(response))
    try:
        candidates = _normalise_candidates(_extract_json_block(text))
    except Exception:
        candidates = _fallback_tickers(text)
    return candidates[: request.candidate_pool_size]


def _decision_score(decision: str, final_state: dict[str, Any]) -> int:
    text = " ".join([
        decision or "",
        final_state.get("final_trade_decision", ""),
        final_state.get("investment_plan", ""),
        final_state.get("narrative_report", ""),
    ]).upper()
    score = 0
    if "MULTI_BAGGER" in text or "2X" in text or "DOUBLE" in text:
        score += 4
    if "BUY" in text:
        score += 3
    if "OVERWEIGHT" in text or "ACCUMULATE" in text:
        score += 2
    if "WATCH" in text or "HOLD" in text:
        score += 1
    if "SELL" in text or "AVOID" in text or "UNDERWEIGHT" in text:
        score -= 3
    return score


def run_discovery(
    request: DiscoveryRequest,
    run_config: ModeRunConfig,
    *,
    selected_analysts: list[str] | None = None,
) -> dict[str, Any]:
    candidates = generate_candidates(request, run_config)
    deep_dive = candidates[: max(1, request.stock_count)]
    results = []
    for candidate in deep_dive:
        ticker = candidate["ticker"]
        focus = (
            "Discovery mode deep dive. User request: "
            f"{request.prompt}\n\nCandidate rationale: {candidate.get('reason', '')}\n"
            f"Potential catalyst: {candidate.get('catalyst', '')}\n"
            f"Main risk: {candidate.get('risk', '')}\n\n"
            "Evaluate whether this has a realistic 3-12 month re-rating or 2x path. "
            "If it is a good company but not a multi-bagger setup, say so clearly."
        )
        final_state, decision = run_single_ticker(
            ticker,
            run_config,
            user_prompt=focus,
            selected_analysts=selected_analysts,
        )
        results.append({
            "candidate": candidate,
            "ticker": ticker,
            "decision": decision,
            "score": _decision_score(decision, final_state),
            "final_state": final_state,
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "mode": "discovery",
        "request": request,
        "candidates": candidates,
        "results": results,
    }
