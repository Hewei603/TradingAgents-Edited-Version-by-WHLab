# Update 0.1: Narrative / Human Lens Analyst

TradingAgents now supports an optional analyst key:

```python
"narrative"
```

This analyst audits the market story rather than mechanically summarizing data.
It focuses on narrative timing, price-in logic, source incentives, retail traps,
virtual-vs-real layers, and what evidence would disconfirm the interpretation.

Example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

ta = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals", "narrative"],
    debug=True,
)

state, decision = ta.propagate(
    "ORCL",
    "2026-06-10",
    user_prompt="""
    Focus on whether Oracle equity issuance / financing fear is already priced in.
    Keep technical analysis light.
    Study narrative reversal risk, Trump/Stargate/OpenAI/AI infrastructure ties,
    and whether retail investors may be overreacting to dilution headlines.
    """,
)

print(state["narrative_report"])
print(decision)
```

CLI:

```powershell
tradingagents analyze --research-focus "Test whether ORCL dilution fear is old news recycled or a real regime shift."
```

The research focus is treated as a hypothesis to test, not as a conclusion to
confirm. The Narrative Analyst must separate confirmed facts, reasonable
inference, speculative possibility, and disconfirming evidence.
