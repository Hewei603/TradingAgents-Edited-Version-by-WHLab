"""TradingAgents Streamlit Web UI"""

import sys
import os
import queue
import threading
import datetime
import time
from pathlib import Path
from collections import deque

# Add project root to path so imports work when run from any directory
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

import streamlit as st

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS
from tradingagents.llm_clients.api_key_env import get_api_key_env
from cli.models import AnalystType, AssetType
from cli.utils import detect_asset_type, filter_analysts_for_asset_type
from cli.stats_handler import StatsCallbackHandler


# ─────────────────────────── constants ────────────────────────────

ANALYST_DISPLAY = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
    "narrative": "Narrative Analyst",
}

ANALYST_ORDER = ["market", "social", "news", "fundamentals", "narrative"]

FIXED_AGENTS = {
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}

ALL_TEAMS_ORDER = {
    "Analyst Team": ["Market Analyst", "Sentiment Analyst", "News Analyst", "Fundamentals Analyst", "Narrative Analyst"],
    **FIXED_AGENTS,
}

PROVIDERS = [
    ("OpenAI", "openai"),
    ("Anthropic", "anthropic"),
    ("Google", "google"),
    ("xAI", "xai"),
    ("DeepSeek", "deepseek"),
    ("Qwen (International)", "qwen"),
    ("Qwen (China)", "qwen-cn"),
    ("GLM (Z.AI International)", "glm"),
    ("GLM (BigModel China)", "glm-cn"),
    ("MiniMax (Global)", "minimax"),
    ("MiniMax (China)", "minimax-cn"),
    ("OpenRouter", "openrouter"),
    ("Azure OpenAI", "azure"),
    ("Ollama", "ollama"),
]

PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/",
    "google": None,
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "glm": "https://api.z.ai/api/paas/v4/",
    "glm-cn": "https://open.bigmodel.cn/api/paas/v4/",
    "minimax": "https://api.minimax.io/v1",
    "minimax-cn": "https://api.minimaxi.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "azure": None,
    "ollama": os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
}

LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean",
    "Hindi", "Spanish", "Portuguese", "French",
    "German", "Arabic", "Russian",
]

REPORT_SECTIONS = {
    "market_report": ("market", "Market Analyst", "Market Analysis"),
    "sentiment_report": ("social", "Sentiment Analyst", "Social Sentiment"),
    "news_report": ("news", "News Analyst", "News Analysis"),
    "fundamentals_report": ("fundamentals", "Fundamentals Analyst", "Fundamentals Analysis"),
    "narrative_report": ("narrative", "Narrative Analyst", "Narrative / Human Lens Analysis"),
    "investment_plan": (None, "Research Manager", "Research Team Decision"),
    "trader_investment_plan": (None, "Trader", "Trading Team Plan"),
    "final_trade_decision": (None, "Portfolio Manager", "Portfolio Management Decision"),
}


# ─────────────────────────── session state helpers ────────────────

def _init_session():
    defaults = {
        "phase": "config",          # "config" | "running" | "results"
        "agent_status": {},
        "messages": deque(maxlen=50),
        "report_sections": {},
        "stats": {"llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0},
        "final_state": None,
        "decision": "",
        "analysis_thread": None,
        "result_queue": None,
        "start_time": None,
        "error": None,
        "selections": None,
        "processed_ids": set(),
        "current_report_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_run_state():
    st.session_state.phase = "running"
    st.session_state.agent_status = {}
    st.session_state.messages = deque(maxlen=50)
    st.session_state.report_sections = {}
    st.session_state.stats = {"llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0}
    st.session_state.final_state = None
    st.session_state.decision = ""
    st.session_state.error = None
    st.session_state.processed_ids = set()
    st.session_state.current_report_text = ""


# ─────────────────────────── model helpers ────────────────────────

def _get_model_list(provider: str, mode: str) -> list[tuple[str, str]]:
    options = MODEL_OPTIONS.get(provider, {}).get(mode, [])
    return [(label, val) for label, val in options if val != "custom"]


def _model_labels(provider, mode):
    pairs = _get_model_list(provider, mode)
    return [label for label, _ in pairs]


def _model_values(provider, mode):
    pairs = _get_model_list(provider, mode)
    return [val for _, val in pairs]


# ─────────────────────────── background analysis ──────────────────

def _run_analysis_bg(selections: dict, result_q: queue.Queue):
    """Background thread: runs the full TradingAgents graph and pushes updates."""
    try:
        mode = selections.get("mode", "single")
        if mode != "single":
            _run_batch_mode_bg(selections, result_q)
            return

        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.graph.analyst_execution import (
            build_analyst_execution_plan,
            AnalystWallTimeTracker,
            sync_analyst_tracker_from_chunk,
            get_initial_analyst_node,
        )
        from cli.main import (
            update_analyst_statuses,
            classify_message_type,
            MessageBuffer,
            ANALYST_ORDER as CLI_ANALYST_ORDER,
        )

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = selections["research_depth"]
        config["max_risk_discuss_rounds"] = selections["research_depth"]
        config["quick_think_llm"] = selections["shallow_thinker"]
        config["deep_think_llm"] = selections["deep_thinker"]
        config["backend_url"] = selections.get("backend_url")
        config["llm_provider"] = selections["llm_provider"]
        config["google_thinking_level"] = selections.get("google_thinking_level")
        config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
        config["anthropic_effort"] = selections.get("anthropic_effort")
        config["output_language"] = selections.get("output_language", "English")

        # Ensure API key is in environment
        api_key = selections.get("api_key", "")
        env_var = get_api_key_env(selections["llm_provider"])
        if api_key and env_var:
            os.environ[env_var] = api_key

        stats_handler = StatsCallbackHandler()

        selected_set = set(selections["analysts"])
        selected_analyst_keys = [a for a in CLI_ANALYST_ORDER if a in selected_set]

        analyst_execution_plan = build_analyst_execution_plan(
            selected_analyst_keys,
            concurrency_limit=config["analyst_concurrency_limit"],
        )
        wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

        graph = TradingAgentsGraph(
            selected_analyst_keys,
            config=config,
            debug=False,
            callbacks=[stats_handler],
        )

        # Build a MessageBuffer for status tracking only (not for display)
        mb = MessageBuffer()
        mb.init_for_analysis(selected_analyst_keys)

        instrument_context = graph.resolve_instrument_context(
            selections["ticker"], selections["asset_type"]
        )
        init_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
            instrument_context=instrument_context,
            user_prompt=selections.get("user_prompt", ""),
        )
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Signal initial analyst as in_progress
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        mb.update_agent_status(first_analyst, "in_progress")
        result_q.put({
            "type": "status",
            "agent_status": dict(mb.agent_status),
            "stats": stats_handler.get_stats(),
        })

        all_chunks = []

        for chunk in graph.graph.stream(init_state, **args):
            if not isinstance(chunk, dict):
                result_q.put({
                    "type": "chunk",
                    "messages": [(
                        datetime.datetime.now().strftime("%H:%M:%S"),
                        "System",
                        f"Ignored unexpected stream chunk: {type(chunk).__name__}",
                    )],
                    "stats": stats_handler.get_stats(),
                })
                continue
            all_chunks.append(chunk)

            # Process messages
            msgs_out = []
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in mb._processed_message_ids:
                        continue
                    mb._processed_message_ids.add(msg_id)
                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    mb.add_message(msg_type, content)
                    msgs_out.append((datetime.datetime.now().strftime("%H:%M:%S"), msg_type, content))
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        name = tc["name"] if isinstance(tc, dict) else tc.name
                        _args = tc["args"] if isinstance(tc, dict) else tc.args
                        arg_str = str(_args)
                        if len(arg_str) > 80:
                            arg_str = arg_str[:77] + "..."
                        mb.add_tool_call(name, _args if isinstance(_args, dict) else {})
                        msgs_out.append((datetime.datetime.now().strftime("%H:%M:%S"), "Tool", f"{name}: {arg_str}"))

            # Update analyst statuses
            update_analyst_statuses(mb, chunk, wall_time_tracker=wall_time_tracker)

            # Research team
            if chunk.get("investment_debate_state"):
                debate = chunk["investment_debate_state"]
                if not isinstance(debate, dict):
                    debate = {}
                bull = debate.get("bull_history", "").strip()
                bear = debate.get("bear_history", "").strip()
                judge = debate.get("judge_decision", "").strip()
                if bull or bear:
                    for ag in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                        mb.update_agent_status(ag, "in_progress")
                if bull:
                    mb.update_report_section("investment_plan", f"### Bull Researcher Analysis\n{bull}")
                if bear:
                    mb.update_report_section("investment_plan", f"### Bear Researcher Analysis\n{bear}")
                if judge:
                    mb.update_report_section("investment_plan", f"### Research Manager Decision\n{judge}")
                    for ag in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                        mb.update_agent_status(ag, "completed")
                    mb.update_agent_status("Trader", "in_progress")

            # Trading team
            if chunk.get("trader_investment_plan"):
                mb.update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
                mb.update_agent_status("Trader", "completed")
                mb.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk management
            if chunk.get("risk_debate_state"):
                risk = chunk["risk_debate_state"]
                if not isinstance(risk, dict):
                    risk = {}
                agg = risk.get("aggressive_history", "").strip()
                con = risk.get("conservative_history", "").strip()
                neu = risk.get("neutral_history", "").strip()
                judge = risk.get("judge_decision", "").strip()
                if agg:
                    mb.update_agent_status("Aggressive Analyst", "in_progress")
                    mb.update_report_section("final_trade_decision", f"### Aggressive Analyst\n{agg}")
                if con:
                    mb.update_agent_status("Conservative Analyst", "in_progress")
                    mb.update_report_section("final_trade_decision", f"### Conservative Analyst\n{con}")
                if neu:
                    mb.update_agent_status("Neutral Analyst", "in_progress")
                    mb.update_report_section("final_trade_decision", f"### Neutral Analyst\n{neu}")
                if judge:
                    mb.update_report_section("final_trade_decision", f"### Portfolio Manager Decision\n{judge}")
                    for ag in ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"]:
                        mb.update_agent_status(ag, "completed")

            result_q.put({
                "type": "chunk",
                "agent_status": dict(mb.agent_status),
                "report_sections": dict(mb.report_sections),
                "current_report": mb.current_report or "",
                "messages": list(msgs_out),
                "stats": stats_handler.get_stats(),
            })

        # Merge final state
        final_state = {}
        for c in all_chunks:
            final_state.update(c)

        decision = graph.process_signal(final_state.get("final_trade_decision", ""))

        result_q.put({
            "type": "done",
            "final_state": final_state,
            "decision": decision,
            "agent_status": dict(mb.agent_status),
            "stats": stats_handler.get_stats(),
        })

    except Exception as e:
        import traceback
        result_q.put({"type": "error", "error": str(e), "traceback": traceback.format_exc()})


def _mode_run_config_from_selections(selections: dict):
    from tradingagents.modes import ModeRunConfig

    return ModeRunConfig(
        trade_date=selections["analysis_date"],
        research_depth=selections["research_depth"],
        llm_provider=selections["llm_provider"],
        quick_model=selections["shallow_thinker"],
        deep_model=selections["deep_thinker"],
        backend_url=selections.get("backend_url"),
        output_language=selections.get("output_language", "English"),
        google_thinking_level=selections.get("google_thinking_level"),
        openai_reasoning_effort=selections.get("openai_reasoning_effort"),
        anthropic_effort=selections.get("anthropic_effort"),
        callbacks=[StatsCallbackHandler()],
    )


def _run_batch_mode_bg(selections: dict, result_q: queue.Queue):
    """Background thread for discovery and portfolio modes."""
    env_var = get_api_key_env(selections["llm_provider"])
    api_key = selections.get("api_key", "")
    if api_key and env_var:
        os.environ[env_var] = api_key

    run_config = _mode_run_config_from_selections(selections)
    selected_analysts = selections.get("analysts") or ["market", "social", "news", "fundamentals", "narrative"]
    mode = selections.get("mode")

    result_q.put({
        "type": "chunk",
        "messages": [(
            datetime.datetime.now().strftime("%H:%M:%S"),
            "System",
            f"Starting {mode} mode. The TradingAgents single-ticker core will be reused for each deep dive.",
        )],
        "agent_status": {"Mode Orchestrator": "in_progress"},
        "stats": {"llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0},
    })

    if mode == "discovery":
        from tradingagents.modes import DiscoveryRequest, run_discovery

        request = DiscoveryRequest(
            prompt=selections["discovery_prompt"],
            stock_count=selections["stock_count"],
            market=selections["market"],
            risk_level=selections["risk_level"],
            timeframe=selections["timeframe"],
            candidate_pool_size=max(selections["stock_count"] * 2, selections.get("candidate_pool_size", 12)),
        )
        result = run_discovery(request, run_config, selected_analysts=selected_analysts)
        decision = f"{len(result['results'])} candidate deep dives completed"

    elif mode == "portfolio":
        from tradingagents.modes import PortfolioRequest, run_portfolio_analysis

        request = PortfolioRequest(
            file_path=selections["portfolio_file_path"],
            account_type=selections["account_type"],
            analysis_goal=selections["portfolio_goal"],
            max_positions_to_analyze=selections["max_positions_to_analyze"],
        )
        result = run_portfolio_analysis(request, run_config, selected_analysts=selected_analysts)
        decision = f"Portfolio risk: {result['portfolio_risk_level']}"

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    result_q.put({
        "type": "done",
        "final_state": {"mode_result": result},
        "decision": decision,
        "agent_status": {"Mode Orchestrator": "completed"},
        "stats": {},
    })


# ─────────────────────────── UI pages ─────────────────────────────

def _page_config():
    """Configuration form page."""
    st.title("TradingAgents")
    st.caption("Multi-Agents LLM Financial Trading Framework")
    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Analysis Settings")

        mode_label = st.radio(
            "Mode",
            [
                "Single Stock Analysis",
                "Find Stocks from Prompt",
                "Portfolio / Margin Account",
            ],
            horizontal=False,
        )
        mode = {
            "Single Stock Analysis": "single",
            "Find Stocks from Prompt": "discovery",
            "Portfolio / Margin Account": "portfolio",
        }[mode_label]

        ticker = ""
        discovery_prompt = ""
        stock_count = 5
        market = "US"
        risk_level = "aggressive"
        timeframe = "3-12 months"
        uploaded_portfolio = None
        account_type = "margin"
        portfolio_goal = ""
        max_positions_to_analyze = 5

        if mode == "single":
            ticker = st.text_input(
                "Ticker Symbol",
                value="SPY",
                placeholder="e.g. SPY, 0700.HK, BTC-USD",
                help="Enter a stock ticker, with exchange suffix if needed.",
            ).strip().upper()

        elif mode == "discovery":
            discovery_prompt = st.text_area(
                "Discovery Prompt",
                value="",
                placeholder="Example: Find 3-12 month 2x candidates in AI infrastructure, power, data centers, or overlooked software turnarounds.",
                height=150,
            )
            stock_count = st.number_input("Number of stocks to deep dive", min_value=1, max_value=10, value=5, step=1)
            market = st.selectbox("Market", ["US", "HK", "China", "Global"], index=0)
            risk_level = st.selectbox("Risk Level", ["aggressive", "balanced", "conservative"], index=0)
            timeframe = st.selectbox("Target Timeframe", ["1-3 months", "3-12 months", "6-18 months"], index=1)

        else:
            uploaded_portfolio = st.file_uploader(
                "Portfolio File",
                type=["csv", "xlsx", "xls"],
                help="Upload a holdings export with at least a ticker/symbol column. CSV/XLSX supported first.",
            )
            account_type = st.selectbox("Account Type", ["margin", "cash"], index=0)
            portfolio_goal = st.text_area(
                "Portfolio Analysis Goal",
                value="Identify concentration risk, margin risk, add/trim candidates, and 3-12 month upside candidates.",
                height=120,
            )
            max_positions_to_analyze = st.number_input(
                "Positions to deep dive",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="The largest positions by market value/weight are analyzed first.",
            )

        analysis_date = st.date_input(
            "Analysis Date",
            value=datetime.date.today(),
            max_value=datetime.date.today(),
        )

        output_language = st.selectbox("Output Language", LANGUAGES, index=0)

        asset_type = "crypto" if mode == "single" and any(ticker.endswith(s) for s in ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")) else "stock"

        available_analysts = ["market", "social", "news", "fundamentals", "narrative"]
        if asset_type == "crypto":
            available_analysts = [a for a in available_analysts if a != "fundamentals"]

        analyst_labels = [ANALYST_DISPLAY[a] for a in available_analysts]
        selected_labels = st.multiselect(
            "Analyst Team",
            options=analyst_labels,
            default=analyst_labels,
            help="Select at least one analyst.",
        )
        selected_analysts = [a for a in available_analysts if ANALYST_DISPLAY[a] in selected_labels]

        user_prompt = ""
        if mode == "single":
            user_prompt = st.text_area(
                "Research Focus",
                value="",
                placeholder="Optional: thesis to test. Leave blank for autonomous Ghost Story Scan.",
                help="Optional hypothesis only; the Narrative Analyst scans market narratives even when blank.",
                height=120,
            )

        depth_options = {"Shallow (1 round)": 1, "Medium (3 rounds)": 3, "Deep (5 rounds)": 5}
        research_depth_label = st.radio("Research Depth", list(depth_options.keys()), index=0)
        research_depth = depth_options[research_depth_label]

    with col_right:
        st.subheader("LLM Settings")

        provider_labels = [label for label, _ in PROVIDERS]
        provider_keys = [key for _, key in PROVIDERS]
        provider_label = st.selectbox("LLM Provider", provider_labels, index=0)
        provider = provider_keys[provider_labels.index(provider_label)]

        # Quick-think model
        quick_labels = _model_labels(provider, "quick")
        quick_values = _model_values(provider, "quick")
        if quick_labels:
            quick_idx = st.selectbox("Quick-Thinking Model", range(len(quick_labels)),
                                     format_func=lambda i: quick_labels[i], index=0)
            shallow_thinker = quick_values[quick_idx]
        else:
            shallow_thinker = st.text_input("Quick-Thinking Model ID", value=DEFAULT_CONFIG["quick_think_llm"])

        # Deep-think model
        deep_labels = _model_labels(provider, "deep")
        deep_values = _model_values(provider, "deep")
        if deep_labels:
            deep_idx = st.selectbox("Deep-Thinking Model", range(len(deep_labels)),
                                    format_func=lambda i: deep_labels[i], index=0)
            deep_thinker = deep_values[deep_idx]
        else:
            deep_thinker = st.text_input("Deep-Thinking Model ID", value=DEFAULT_CONFIG["deep_think_llm"])

        # Provider-specific options
        google_thinking = None
        openai_effort = None
        anthropic_effort = None

        if provider == "google":
            thinking_choice = st.selectbox(
                "Gemini Thinking Mode",
                ["Enable Thinking (recommended)", "Minimal/Disable Thinking"],
            )
            google_thinking = "high" if "Enable" in thinking_choice else "minimal"

        elif provider == "openai":
            effort_choice = st.selectbox(
                "OpenAI Reasoning Effort",
                ["Medium (Default)", "High (More thorough)", "Low (Faster)"],
            )
            openai_effort = effort_choice.split()[0].lower()

        elif provider == "anthropic":
            effort_choice = st.selectbox(
                "Claude Effort Level",
                ["High (recommended)", "Medium (balanced)", "Low (faster, cheaper)"],
            )
            anthropic_effort = effort_choice.split()[0].lower()

        # API key
        env_var = get_api_key_env(provider)
        api_key = ""
        if env_var:
            default_key = os.environ.get(env_var, "")
            api_key = st.text_input(
                f"API Key ({env_var})",
                value=default_key,
                type="password",
                help=f"Will be set as {env_var} for this session.",
            )

    st.divider()

    # Validation and submit
    errors = []
    if mode == "single" and not ticker:
        errors.append("Ticker symbol is required.")
    if mode == "discovery" and not discovery_prompt.strip():
        errors.append("Discovery prompt is required.")
    if mode == "portfolio" and uploaded_portfolio is None:
        errors.append("Upload a CSV or Excel portfolio file.")
    if not selected_analysts:
        errors.append("Select at least one analyst.")
    if env_var and not api_key and provider != "ollama":
        errors.append(f"{env_var} is required for {provider_label}.")

    if errors:
        for e in errors:
            st.warning(e)

    if st.button("Start Analysis", type="primary", disabled=bool(errors)):
        portfolio_file_path = ""
        if mode == "portfolio" and uploaded_portfolio is not None:
            upload_dir = _ROOT / "run_outputs" / "portfolio_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(uploaded_portfolio.name).name}"
            portfolio_file_path = str(upload_dir / safe_name)
            with open(portfolio_file_path, "wb") as f:
                f.write(uploaded_portfolio.getbuffer())

        selections = {
            "mode": mode,
            "ticker": ticker,
            "asset_type": asset_type,
            "analysis_date": str(analysis_date),
            "analysts": selected_analysts,
            "research_depth": research_depth,
            "llm_provider": provider,
            "backend_url": PROVIDER_URLS.get(provider),
            "shallow_thinker": shallow_thinker,
            "deep_thinker": deep_thinker,
            "google_thinking_level": google_thinking,
            "openai_reasoning_effort": openai_effort,
            "anthropic_effort": anthropic_effort,
            "output_language": output_language,
            "api_key": api_key,
            "user_prompt": user_prompt,
            "discovery_prompt": discovery_prompt,
            "stock_count": int(stock_count),
            "market": market,
            "risk_level": risk_level,
            "timeframe": timeframe,
            "portfolio_file_path": portfolio_file_path,
            "account_type": account_type,
            "portfolio_goal": portfolio_goal,
            "max_positions_to_analyze": int(max_positions_to_analyze),
        }
        _reset_run_state()

        # Init agent_status
        agent_status = {}
        for a in selected_analysts:
            agent_status[ANALYST_DISPLAY[a]] = "pending"
        for agents in FIXED_AGENTS.values():
            for ag in agents:
                agent_status[ag] = "pending"
        st.session_state.agent_status = agent_status

        # Init report sections
        report_sections = {}
        for sec, (analyst_key, _, _) in REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in selected_analysts:
                report_sections[sec] = None
        st.session_state.report_sections = report_sections

        st.session_state.selections = selections
        st.session_state.start_time = time.time()

        result_q = queue.Queue()
        st.session_state.result_queue = result_q

        t = threading.Thread(
            target=_run_analysis_bg,
            args=(selections, result_q),
            daemon=True,
        )
        t.start()
        st.session_state.analysis_thread = t
        st.rerun()


def _status_icon(status: str) -> str:
    return {"pending": "🟡", "in_progress": "⏳", "completed": "✅", "error": "❌"}.get(status, "❓")


def _page_running():
    """Live analysis progress page."""
    sel = st.session_state.selections or {}
    mode = sel.get("mode", "single")
    ticker = sel.get("ticker", "")
    analysis_date = sel.get("analysis_date", "")

    if mode == "discovery":
        st.title(f"Finding Stocks — {analysis_date}")
    elif mode == "portfolio":
        st.title(f"Analyzing Portfolio — {analysis_date}")
    else:
        st.title(f"Analyzing {ticker} — {analysis_date}")

    # Drain the result queue
    result_q: queue.Queue = st.session_state.result_queue
    is_done = False

    while True:
        try:
            msg = result_q.get_nowait()
        except queue.Empty:
            break

        if msg["type"] == "error":
            st.session_state.error = msg["error"]
            st.session_state.traceback = msg.get("traceback", "")
            st.session_state.phase = "results"
            st.rerun()
            return

        if msg["type"] in ("chunk", "status"):
            st.session_state.agent_status.update(msg.get("agent_status", {}))
            for sec, content in msg.get("report_sections", {}).items():
                if content is not None:
                    st.session_state.report_sections[sec] = content
            st.session_state.stats.update(msg.get("stats", {}))
            if msg.get("current_report"):
                st.session_state.current_report_text = msg["current_report"]
            for m in msg.get("messages", []):
                st.session_state.messages.append(m)

        if msg["type"] == "done":
            st.session_state.final_state = msg.get("final_state")
            st.session_state.decision = msg.get("decision", "")
            st.session_state.agent_status.update(msg.get("agent_status", {}))
            st.session_state.stats.update(msg.get("stats", {}))
            for ag in st.session_state.agent_status:
                st.session_state.agent_status[ag] = "completed"
            is_done = True
            break

    # ── layout ──
    col_status, col_msgs = st.columns([1, 2])

    with col_status:
        st.subheader("Agent Status")
        rows = []
        for team, agents in ALL_TEAMS_ORDER.items():
            active = [a for a in agents if a in st.session_state.agent_status]
            if not active:
                continue
            for i, ag in enumerate(active):
                status = st.session_state.agent_status.get(ag, "pending")
                rows.append({
                    "Team": team if i == 0 else "",
                    "Agent": ag,
                    "Status": f"{_status_icon(status)} {status}",
                })
        if rows:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                height=min(40 + len(rows) * 35, 500),
            )

    with col_msgs:
        st.subheader("Messages & Tools")
        msgs = list(st.session_state.messages)[-20:]
        if msgs:
            import pandas as pd
            df = pd.DataFrame(msgs, columns=["Time", "Type", "Content"])
            df["Content"] = df["Content"].str[:120]
            st.dataframe(df[::-1].reset_index(drop=True), use_container_width=True,
                         hide_index=True, height=min(40 + len(df) * 35, 460))
        else:
            st.info("Waiting for messages...")

    # Current report preview
    report_text = st.session_state.get("current_report_text", "")
    if report_text:
        with st.expander("Current Report (latest section)", expanded=True):
            st.markdown(report_text)
    else:
        st.info("Waiting for analysis report...")

    # Stats footer
    stats = st.session_state.stats
    elapsed = time.time() - (st.session_state.start_time or time.time())
    elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    agents_done = sum(1 for s in st.session_state.agent_status.values() if s == "completed")
    agents_total = len(st.session_state.agent_status)
    reports_done = sum(
        1 for sec, content in st.session_state.report_sections.items()
        if content is not None
    )
    reports_total = len(st.session_state.report_sections)

    cols = st.columns(6)
    cols[0].metric("Agents", f"{agents_done}/{agents_total}")
    cols[1].metric("LLM Calls", stats.get("llm_calls", 0))
    cols[2].metric("Tool Calls", stats.get("tool_calls", 0))
    tok_in = stats.get("tokens_in", 0)
    tok_out = stats.get("tokens_out", 0)
    cols[3].metric("Tokens ↑", f"{tok_in/1000:.1f}k" if tok_in >= 1000 else str(tok_in))
    cols[4].metric("Reports", f"{reports_done}/{reports_total}")
    cols[5].metric("Elapsed", elapsed_str)

    if is_done:
        st.session_state.phase = "results"
        st.rerun()
    else:
        time.sleep(0.5)
        st.rerun()


def _build_full_report(final_state: dict, report_sections: dict) -> str:
    """Build a complete Markdown report string from final_state."""
    parts = []

    # I. Analyst Team
    analyst_parts = []
    if final_state.get("market_report"):
        analyst_parts.append(f"### Market Analyst\n{final_state['market_report']}")
    if final_state.get("sentiment_report"):
        analyst_parts.append(f"### Sentiment Analyst\n{final_state['sentiment_report']}")
    if final_state.get("news_report"):
        analyst_parts.append(f"### News Analyst\n{final_state['news_report']}")
    if final_state.get("fundamentals_report"):
        analyst_parts.append(f"### Fundamentals Analyst\n{final_state['fundamentals_report']}")
    if final_state.get("narrative_report"):
        analyst_parts.append(f"### Narrative Analyst\n{final_state['narrative_report']}")
    if analyst_parts:
        parts.append("## I. Analyst Team Reports\n\n" + "\n\n".join(analyst_parts))

    # II. Research Team
    debate = final_state.get("investment_debate_state") or {}
    research_parts = []
    if debate.get("bull_history"):
        research_parts.append(f"### Bull Researcher\n{debate['bull_history']}")
    if debate.get("bear_history"):
        research_parts.append(f"### Bear Researcher\n{debate['bear_history']}")
    if debate.get("judge_decision"):
        research_parts.append(f"### Research Manager\n{debate['judge_decision']}")
    if research_parts:
        parts.append("## II. Research Team Decision\n\n" + "\n\n".join(research_parts))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        parts.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # IV. Risk Management
    risk = final_state.get("risk_debate_state") or {}
    risk_parts = []
    if risk.get("aggressive_history"):
        risk_parts.append(f"### Aggressive Analyst\n{risk['aggressive_history']}")
    if risk.get("conservative_history"):
        risk_parts.append(f"### Conservative Analyst\n{risk['conservative_history']}")
    if risk.get("neutral_history"):
        risk_parts.append(f"### Neutral Analyst\n{risk['neutral_history']}")
    if risk_parts:
        parts.append("## IV. Risk Management\n\n" + "\n\n".join(risk_parts))

    # V. Portfolio Manager
    if risk.get("judge_decision"):
        parts.append(f"## V. Portfolio Manager Decision\n\n{risk['judge_decision']}")

    return "\n\n---\n\n".join(parts)


def _single_state_summary(ticker: str, decision: str, final_state: dict) -> str:
    risk = final_state.get("risk_debate_state") or {}
    pm = risk.get("judge_decision") or final_state.get("final_trade_decision", "")
    narrative = final_state.get("narrative_report", "")
    parts = [
        f"## {ticker}",
        f"**Decision:** {decision}",
    ]
    if pm:
        parts.append(f"### Portfolio Manager\n{pm}")
    if narrative:
        parts.append(f"### Narrative / Human Lens\n{narrative}")
    return "\n\n".join(parts)


def _render_discovery_results(mode_result: dict, analysis_date: str):
    request = mode_result.get("request")
    results = mode_result.get("results", [])
    candidates = mode_result.get("candidates", [])

    st.title("Stock Discovery Results")
    st.caption(f"Analysis date: {analysis_date}")
    if request:
        st.markdown(f"**Request:** {request.prompt}")

    st.subheader("Candidate Pool")
    if candidates:
        import pandas as pd
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)

    st.subheader("Deep Dive Ranking")
    if results:
        import pandas as pd
        rows = []
        for idx, item in enumerate(results, start=1):
            candidate = item.get("candidate", {})
            rows.append({
                "Rank": idx,
                "Ticker": item.get("ticker"),
                "Decision": item.get("decision"),
                "Score": item.get("score"),
                "Reason": candidate.get("reason", ""),
                "Catalyst": candidate.get("catalyst", ""),
                "Risk": candidate.get("risk", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for item in results:
            ticker = item.get("ticker", "")
            with st.expander(f"{ticker} deep dive", expanded=False):
                st.markdown(_single_state_summary(ticker, item.get("decision", ""), item.get("final_state", {})))

    report_md = "# Stock Discovery Report\n\n" + "\n\n---\n\n".join(
        _single_state_summary(item.get("ticker", ""), item.get("decision", ""), item.get("final_state", {}))
        for item in results
    )
    st.download_button(
        "Download Discovery Report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name=f"discovery_{analysis_date}_report.md",
        mime="text/markdown",
    )


def _render_portfolio_results(mode_result: dict, analysis_date: str):
    summary = mode_result.get("summary", {})
    results = mode_result.get("position_results", [])

    st.title("Portfolio / Margin Account Results")
    st.caption(f"Analysis date: {analysis_date}")

    cols = st.columns(5)
    cols[0].metric("Risk", mode_result.get("portfolio_risk_level", "n/a"))
    cols[1].metric("Positions", summary.get("position_count", 0))
    total = summary.get("total_market_value")
    cols[2].metric("Market Value", f"{total:,.0f}" if total else "n/a")
    cols[3].metric("Top Position", f"{summary.get('top_position_weight', 0):.1f}%")
    cols[4].metric("Top 5", f"{summary.get('top_5_weight', 0):.1f}%")

    positions = summary.get("positions", [])
    if positions:
        import pandas as pd
        st.subheader("Parsed Holdings")
        st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)

    st.subheader("Position Deep Dives")
    for item in results:
        ticker = item.get("ticker", "")
        position = item.get("position", {})
        title = f"{ticker} — {item.get('decision', '')}"
        with st.expander(title, expanded=False):
            st.json(position)
            st.markdown(_single_state_summary(ticker, item.get("decision", ""), item.get("final_state", {})))

    report_md = "# Portfolio Analysis Report\n\n"
    report_md += f"Portfolio risk: {mode_result.get('portfolio_risk_level', 'n/a')}\n\n"
    report_md += "\n\n---\n\n".join(
        _single_state_summary(item.get("ticker", ""), item.get("decision", ""), item.get("final_state", {}))
        for item in results
    )
    st.download_button(
        "Download Portfolio Report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name=f"portfolio_{analysis_date}_report.md",
        mime="text/markdown",
    )


def _page_results():
    """Final results page."""
    if st.session_state.error:
        st.error("Analysis failed")
        st.code(st.session_state.error)
        if st.button("Back to Configuration"):
            st.session_state.phase = "config"
            st.rerun()
        return

    sel = st.session_state.selections or {}
    final_state = st.session_state.final_state or {}
    decision = st.session_state.decision or ""
    mode = sel.get("mode", "single")

    ticker = sel.get("ticker", "")
    analysis_date = sel.get("analysis_date", "")

    if mode == "discovery":
        _render_discovery_results(final_state.get("mode_result", {}), analysis_date)
        if st.button("New Analysis"):
            st.session_state.phase = "config"
            st.rerun()
        return

    if mode == "portfolio":
        _render_portfolio_results(final_state.get("mode_result", {}), analysis_date)
        if st.button("New Analysis"):
            st.session_state.phase = "config"
            st.rerun()
        return

    st.title(f"Analysis Report: {ticker}")
    st.caption(f"Analysis date: {analysis_date}")

    # Final decision highlight
    decision_upper = decision.upper()
    if "BUY" in decision_upper:
        st.success(f"Final Decision: {decision}", icon="📈")
    elif "SELL" in decision_upper:
        st.error(f"Final Decision: {decision}", icon="📉")
    else:
        st.warning(f"Final Decision: {decision}", icon="⚖️")

    st.divider()

    # I. Analyst Team
    analyst_data = [
        ("Market Analysis", final_state.get("market_report")),
        ("Social Sentiment", final_state.get("sentiment_report")),
        ("News Analysis", final_state.get("news_report")),
        ("Fundamentals Analysis", final_state.get("fundamentals_report")),
        ("Narrative / Human Lens Analysis", final_state.get("narrative_report")),
    ]
    analyst_filled = [(t, c) for t, c in analyst_data if c]
    if analyst_filled:
        st.subheader("I. Analyst Team Reports")
        for title, content in analyst_filled:
            with st.expander(title, expanded=False):
                st.markdown(content)

    # II. Research Team
    debate = final_state.get("investment_debate_state") or {}
    research_data = [
        ("Bull Researcher", debate.get("bull_history")),
        ("Bear Researcher", debate.get("bear_history")),
        ("Research Manager Decision", debate.get("judge_decision")),
    ]
    research_filled = [(t, c) for t, c in research_data if c]
    if research_filled:
        st.subheader("II. Research Team Decision")
        for title, content in research_filled:
            with st.expander(title, expanded=False):
                st.markdown(content)

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        st.subheader("III.ing Team Plan")
        with st.expander("Trader", expanded=False):
            st.markdown(final_state["trader_investment_plan"])

    # IV. Risk Management
    risk = final_state.get("risk_debate_state") or {}
    risk_data = [
        ("Aggressive Analyst", risk.get("aggressive_history")),
        ("Conservative Analyst", risk.get("conservative_history")),
        ("Neutral Analyst", risk.get("neutral_history")),
    ]
    risk_filled = [(t, c) for t, c in risk_data if c]
    if risk_filled:
        st.subheader("IV. Risk Management")
        for title, content in risk_filled:
            with st.expander(title, expanded=False):
                st.markdown(content)

    # V. Portfolio Manager
    if risk.get("judge_decision"):
        st.subheader("V. Portfolio Manager Decision")
        with st.expander("Portfolio Manager", expanded=True):
            st.markdown(risk["judge_decision"])

    st.divider()

    # Stats
    stats = st.session_state.stats
    elapsed = time.time() - (st.session_state.start_time or time.time())
    cols = st.columns(5)
    cols[0].metric("LLM Calls", stats.get("llm_calls", 0))
    cols[1].metric("Tool Calls", stats.get("tool_calls", 0))
    cols[2].metric("Tokens In", f"{stats.get('tokens_in', 0) / 1000:.1f}k")
    cols[3].metric("Tokens Out", f"{stats.get('tokens_out', 0) / 1000:.1f}k")
    cols[4].metric("Total Time", f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

    st.divider()

    # Download
    full_report = _build_full_report(final_state, st.session_state.report_sections)
    header = f"# Trading Analysis Report: {ticker}\n\nDate: {analysis_date}  \nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    report_md = header + full_report

    st.download_button(
        "Download Full Report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name=f"{ticker}_{analysis_date}_report.md",
        mime="text/markdown",
    )

    if st.button("New Analysis"):
        st.session_state.phase = "config"
        st.rerun()


# ─────────────────────────── main ─────────────────────────────────

def main():
    st.set_page_config(
        page_title="TradingAgents",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_session()

    phase = st.session_state.phase
    if phase == "config":
        _page_config()
    elif phase == "running":
        _page_running()
    elif phase == "results":
        _page_results()


if __name__ == "__main__":
    main()
