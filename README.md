# TradingAgents-Edited

TradingAgents is a multi-agent LLM research framework for financial-market analysis. It coordinates analyst, researcher, trader, risk-management, and portfolio-manager agents to produce a structured trading thesis from market data, fundamentals, news, sentiment, and optional user research focus.

This project is for research and education only. It is not financial, investment, or trading advice, and model output should not be used as an automated trading signal without independent validation.

## Highlights

- Multi-agent workflow built on LangGraph.
- CLI and Streamlit Web UI.
- Analyst coverage for market/technical data, fundamentals, news, social sentiment, and narrative/human-lens research.
- Multiple LLM providers: OpenAI, Anthropic, Google Gemini, xAI, DeepSeek, Qwen, GLM, MiniMax, OpenRouter, Azure OpenAI, and Ollama.
- Yahoo Finance first data flow with Alpha Vantage fallback support.
- US, non-US, and crypto ticker handling through Yahoo Finance symbols.
- Optional checkpoint resume for interrupted runs.
- Persistent decision memory and reflection across completed analyses.

## Attribution

The core multi-agent trading architecture is derived from the open-source `TauricResearch/TradingAgents` project and has been substantially modified in this repository. The original work is licensed under Apache License 2.0; this repository keeps the Apache-2.0 license and preserves attribution through the license and this notice.

If you use the original TradingAgents research framework in academic work, cite the upstream paper:

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138},
}
```

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python -m venv .venv
.venv\Scripts\activate
pip install .
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

You can also install with `uv`:

```bash
uv sync
```

## Configuration

Copy the example environment file and add only the keys you need:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Supported API key environment variables include:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...
DEEPSEEK_API_KEY=...
DASHSCOPE_API_KEY=...
DASHSCOPE_CN_API_KEY=...
ZHIPU_API_KEY=...
ZHIPU_CN_API_KEY=...
MINIMAX_API_KEY=...
MINIMAX_CN_API_KEY=...
OPENROUTER_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
```

Common runtime overrides:

```bash
TRADINGAGENTS_LLM_PROVIDER=openai
TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5
TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.4-mini
TRADINGAGENTS_OUTPUT_LANGUAGE=English
TRADINGAGENTS_TEMPERATURE=0
TRADINGAGENTS_CHECKPOINT_ENABLED=false
TRADINGAGENTS_RESULTS_DIR=reports
TRADINGAGENTS_CACHE_DIR=.cache/tradingagents
```

For enterprise providers such as Azure OpenAI, copy `.env.enterprise.example` to `.env.enterprise` and fill in provider-specific settings.

## CLI Usage

Run the interactive CLI:

```bash
tradingagents analyze
```

Or run from source:

```bash
python -m cli.main analyze
```

Useful options:

```bash
tradingagents analyze --checkpoint
tradingagents analyze --clear-checkpoints
tradingagents analyze --user-prompt "Focus on AI capex risk and next-quarter guidance"
```

The CLI lets you choose ticker, date, provider, model, depth, analysts, and output language. Completed reports can be saved under `reports/`.

## Web UI

Start the Streamlit UI:

```bash
streamlit run webui/app.py
```

On Windows you can also run:

```bash
start_webui.bat
```

The Web UI reads `.env` from the project root.

## Python Usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5.5"
config["quick_think_llm"] = "gpt-5.4-mini"
config["max_debate_rounds"] = 1

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

## Markets and Symbols

Use Yahoo Finance symbols:

- US equities and ETFs: `AAPL`, `NVDA`, `SPY`
- Hong Kong: `0700.HK`
- Tokyo: `7203.T`
- London: `AZN.L`
- India: `RELIANCE.NS`, `RELIANCE.BO`
- Canada: `SHOP.TO`
- Australia: `BHP.AX`
- China A-shares: `600519.SS`, `000001.SZ`
- Crypto: `BTC-USD`, `ETH-USD`

Regional benchmarks are selected automatically where configured; US tickers default to `SPY`.

## Persistence

Decision memory is written to `~/.tradingagents/memory/trading_memory.md` by default. Override it with:

```bash
TRADINGAGENTS_MEMORY_LOG_PATH=...
```

Checkpoint resume is opt-in with `--checkpoint` or `TRADINGAGENTS_CHECKPOINT_ENABLED=true`. Checkpoints are stored under `~/.tradingagents/cache/checkpoints/` unless `TRADINGAGENTS_CACHE_DIR` is set.

## Reproducibility

LLM analysis is not byte-stable. Even with the same ticker and date, output may vary because providers sample responses and live news/social feeds change over time. To reduce variance, use a low temperature and a model that honors temperature:

```python
config["temperature"] = 0.0
config["deep_think_llm"] = "gpt-4.1"
config["quick_think_llm"] = "gpt-4.1"
```

Historical price windows are date-pinned, but live text sources still reflect the time of collection.

## Development

Run tests:

```bash
pytest
```

Recommended before publishing:

```bash
git status --short
pytest
```

Do not commit `.env`, `.env.enterprise`, local caches, generated reports, or `run_outputs/`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
