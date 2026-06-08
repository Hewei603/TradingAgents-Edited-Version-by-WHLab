import unittest

from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup


class _DummyLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, *_args, **_kwargs):
        raise AssertionError("smoke test should compile, not execute LLM")


class NarrativeAnalystWiringTests(unittest.TestCase):
    def test_initial_state_contains_narrative_and_user_focus(self):
        state = Propagator().create_initial_state(
            "ORCL",
            "2026-06-10",
            user_prompt="Test whether dilution fear is priced in.",
        )

        self.assertEqual(state["user_prompt"], "Test whether dilution fear is priced in.")
        self.assertEqual(state["research_focus"], "Test whether dilution fear is priced in.")
        self.assertEqual(state["narrative_report"], "")

    def test_graph_compiles_with_narrative_analyst(self):
        tool_nodes = {
            "market": ToolNode([get_news]),
            "social": ToolNode([get_news]),
            "news": ToolNode([get_news]),
            "fundamentals": ToolNode([get_news]),
            "narrative": ToolNode([get_news]),
        }
        setup = GraphSetup(
            _DummyLLM(),
            _DummyLLM(),
            tool_nodes,
            ConditionalLogic(),
        )

        workflow = setup.setup_graph(["market", "social", "news", "fundamentals", "narrative"])
        graph = workflow.compile()

        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
