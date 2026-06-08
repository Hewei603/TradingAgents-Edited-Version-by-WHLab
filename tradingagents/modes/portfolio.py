"""Portfolio and margin-account analysis mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.modes.common import ModeRunConfig, run_single_ticker


@dataclass
class PortfolioRequest:
    file_path: str
    account_type: str = "margin"
    analysis_goal: str = "Identify portfolio risk, add/trim candidates, and 3-12 month upside candidates."
    max_positions_to_analyze: int = 5


_COLUMN_ALIASES = {
    "ticker": {
        "ticker", "symbol", "security", "代码", "股票代码", "证券代码",
    },
    "quantity": {
        "quantity", "qty", "shares", "position", "持仓", "数量", "股数",
    },
    "cost_basis": {
        "cost_basis", "average_cost", "avg_cost", "cost", "成本", "平均成本",
    },
    "market_value": {
        "market_value", "value", "marketvalue", "mkt_value", "市值", "市场价值",
    },
    "unrealized_pnl": {
        "unrealized_pnl", "unrealized", "pnl", "gain_loss", "浮动盈亏", "未实现盈亏",
    },
    "weight": {
        "weight", "portfolio_weight", "% of account", "占比", "仓位占比",
    },
}


def _normalise_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    normalised = {_normalise_col(col): col for col in df.columns}
    for target, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalise_col(alias)
            if key in normalised:
                rename[normalised[key]] = target
                break
    return df.rename(columns=rename)


def _to_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_portfolio_file(file_path: str) -> dict[str, Any]:
    path = Path(file_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = _map_columns(df)
    if "ticker" not in df.columns:
        raise ValueError("Portfolio file must include a ticker/symbol column.")

    positions = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker or ticker in {"NAN", "NONE"}:
            continue
        position = {
            "ticker": ticker,
            "quantity": _to_number(row.get("quantity")),
            "cost_basis": _to_number(row.get("cost_basis")),
            "market_value": _to_number(row.get("market_value")),
            "unrealized_pnl": _to_number(row.get("unrealized_pnl")),
            "weight": _to_number(row.get("weight")),
        }
        positions.append(position)

    total_value = sum(p.get("market_value") or 0 for p in positions)
    if total_value > 0:
        for position in positions:
            if position.get("weight") is None and position.get("market_value") is not None:
                position["weight"] = 100 * position["market_value"] / total_value

    positions.sort(key=lambda item: item.get("market_value") or item.get("weight") or 0, reverse=True)
    top_weight = max((p.get("weight") or 0 for p in positions), default=0)
    top_5_weight = sum((p.get("weight") or 0 for p in positions[:5]))
    return {
        "file_path": str(path),
        "positions": positions,
        "position_count": len(positions),
        "total_market_value": total_value or None,
        "top_position_weight": top_weight,
        "top_5_weight": top_5_weight,
    }


def _portfolio_risk_label(summary: dict[str, Any], account_type: str) -> str:
    top_weight = summary.get("top_position_weight") or 0
    top_5 = summary.get("top_5_weight") or 0
    if account_type == "margin" and (top_weight >= 25 or top_5 >= 75):
        return "high"
    if top_weight >= 35 or top_5 >= 85:
        return "high"
    if top_weight >= 20 or top_5 >= 65:
        return "medium"
    return "low"


def run_portfolio_analysis(
    request: PortfolioRequest,
    run_config: ModeRunConfig,
    *,
    selected_analysts: list[str] | None = None,
) -> dict[str, Any]:
    summary = parse_portfolio_file(request.file_path)
    positions = summary["positions"][: max(1, request.max_positions_to_analyze)]
    position_results = []
    for position in positions:
        ticker = position["ticker"]
        focus = (
            f"Portfolio mode deep dive for a {request.account_type} account.\n"
            f"User goal: {request.analysis_goal}\n"
            f"Position context: quantity={position.get('quantity')}, "
            f"cost_basis={position.get('cost_basis')}, market_value={position.get('market_value')}, "
            f"weight_pct={position.get('weight')}, unrealized_pnl={position.get('unrealized_pnl')}.\n\n"
            "Assess whether this position should be held, trimmed, added, hedged, or reassessed. "
            "For margin accounts, prioritize concentration risk, drawdown risk, and forced-selling risk. "
            "Also judge whether it has a realistic 3-12 month re-rating or 2x path."
        )
        final_state, decision = run_single_ticker(
            ticker,
            run_config,
            user_prompt=focus,
            selected_analysts=selected_analysts,
        )
        position_results.append({
            "position": position,
            "ticker": ticker,
            "decision": decision,
            "final_state": final_state,
        })

    return {
        "mode": "portfolio",
        "request": request,
        "summary": summary,
        "portfolio_risk_level": _portfolio_risk_label(summary, request.account_type),
        "position_results": position_results,
    }
