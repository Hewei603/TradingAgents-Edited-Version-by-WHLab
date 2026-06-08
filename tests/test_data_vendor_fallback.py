import copy
import unittest
from unittest import mock

import tradingagents.default_config as default_config
import tradingagents.dataflows.config as dataflows_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config


class DataVendorFallbackTests(unittest.TestCase):
    def setUp(self):
        dataflows_config._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
        set_config({"tool_vendors": {"get_stock_data": "yfinance,alpha_vantage"}})

    def test_default_core_stock_api_falls_back_to_alpha_vantage(self):
        calls = []

        def yfinance_fails(*args, **kwargs):
            calls.append("yfinance")
            raise RuntimeError("Yahoo rate limited")

        def alpha_succeeds(*args, **kwargs):
            calls.append("alpha_vantage")
            return "alpha data"

        patched = {
            "yfinance": yfinance_fails,
            "alpha_vantage": alpha_succeeds,
        }

        with mock.patch.dict(
            interface.VENDOR_METHODS, {"get_stock_data": patched}, clear=False
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "ORCL", "2025-05-27", "2025-06-03"
            )

        self.assertEqual(result, "alpha data")
        self.assertEqual(calls, ["yfinance", "alpha_vantage"])


if __name__ == "__main__":
    unittest.main()
