import types
import unittest
from unittest import mock

from tradingagents.dataflows import yfinance_config


class YFinanceConfigTests(unittest.TestCase):
    def setUp(self):
        self._patchers = []
        for name in (
            "TRADINGAGENTS_YFINANCE_PROXY",
            "YFINANCE_PROXY",
            "TRADINGAGENTS_YFINANCE_HTTP_PROXY",
            "TRADINGAGENTS_YFINANCE_HTTPS_PROXY",
        ):
            patcher = mock.patch.dict("os.environ", {name: ""})
            patcher.start()
            self._patchers.append(patcher)
            yfinance_config.os.environ.pop(name, None)

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()

    def test_default_keeps_legacy_local_proxy(self):
        self.assertEqual(
            yfinance_config.get_yfinance_proxies(),
            {
                "http": yfinance_config.LEGACY_YFINANCE_PROXY,
                "https": yfinance_config.LEGACY_YFINANCE_PROXY,
            },
        )

    def test_shared_proxy_env_overrides_default(self):
        with mock.patch.dict(
            "os.environ",
            {"TRADINGAGENTS_YFINANCE_PROXY": "http://127.0.0.1:7890"},
        ):
            self.assertEqual(
                yfinance_config.get_yfinance_proxies(),
                {
                    "http": "http://127.0.0.1:7890",
                    "https": "http://127.0.0.1:7890",
                },
            )

    def test_shared_proxy_env_can_disable_proxy(self):
        with mock.patch.dict("os.environ", {"TRADINGAGENTS_YFINANCE_PROXY": "off"}):
            self.assertIsNone(yfinance_config.get_yfinance_proxies())

    def test_configure_modern_yfinance_network_proxy(self):
        yf_module = types.SimpleNamespace(
            config=types.SimpleNamespace(network=types.SimpleNamespace(proxy="old"))
        )

        yfinance_config.configure_yfinance_proxy(yf_module)

        self.assertEqual(
            yf_module.config.network.proxy,
            {
                "http": yfinance_config.LEGACY_YFINANCE_PROXY,
                "https": yfinance_config.LEGACY_YFINANCE_PROXY,
            },
        )

    def test_configure_legacy_yfinance_set_config(self):
        calls = []
        yf_module = types.SimpleNamespace(set_config=lambda **kwargs: calls.append(kwargs))

        yfinance_config.configure_yfinance_proxy(yf_module)

        self.assertEqual(
            calls,
            [
                {
                    "proxy": {
                        "http": yfinance_config.LEGACY_YFINANCE_PROXY,
                        "https": yfinance_config.LEGACY_YFINANCE_PROXY,
                    }
                }
            ],
        )
