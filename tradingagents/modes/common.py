"""Shared helpers for higher-level product modes.

These helpers keep the existing TradingAgentsGraph as the analysis kernel.
Discovery and portfolio modes should orchestrate around that kernel instead of
forking analyst logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


DEFAULT_ANALYSTS = ["market", "social", "news", "fundamentals", "narrative"]


@dataclass
class ModeRunConfig:
    trade_date: str
    research_depth: int = 1
    llm_provider: str = "openai"
    quick_model: str | None = None
    deep_model: str | None = None
    backend_url: str | None = None
    output_language: str = "English"
    google_thinking_level: str | None = None
    openai_reasoning_effort: str | None = None
    anthropic_effort: str | None = None
    callbacks: list[Any] | None = None


def build_graph_config(run_config: ModeRunConfig) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = run_config.research_depth
    config["max_risk_discuss_rounds"] = run_config.research_depth
    config["llm_provider"] = run_config.llm_provider
    config["backend_url"] = run_config.backend_url
    config["output_language"] = run_config.output_language
    config["google_thinking_level"] = run_config.google_thinking_level
    config["openai_reasoning_effort"] = run_config.openai_reasoning_effort
    config["anthropic_effort"] = run_config.anthropic_effort
    config["checkpoint_enabled"] = False
    if run_config.quick_model:
        config["quick_think_llm"] = run_config.quick_model
    if run_config.deep_model:
        config["deep_think_llm"] = run_config.deep_model
    return config


def run_single_ticker(
    ticker: str,
    run_config: ModeRunConfig,
    *,
    asset_type: str = "stock",
    user_prompt: str = "",
    selected_analysts: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    graph = TradingAgentsGraph(
        selected_analysts or DEFAULT_ANALYSTS,
        debug=False,
        config=build_graph_config(run_config),
        callbacks=run_config.callbacks,
    )
    final_state, decision = graph.propagate(
        ticker.upper().strip(),
        run_config.trade_date,
        asset_type=asset_type,
        user_prompt=user_prompt,
    )
    return final_state, decision
