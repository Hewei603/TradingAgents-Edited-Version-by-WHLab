import os


LEGACY_YFINANCE_PROXY = "http://127.0.0.1:29290"


def _disabled(value: str) -> bool:
    return value.strip().lower() in {"", "0", "false", "off", "none", "no"}


def get_yfinance_proxies():
    """Return yfinance proxy settings from env, preserving the old local default.

    Set TRADINGAGENTS_YFINANCE_PROXY to a proxy URL such as
    ``http://127.0.0.1:7890``. Set it to ``off``/``none``/``0`` to disable
    proxying. Separate HTTP/HTTPS values can be provided with
    TRADINGAGENTS_YFINANCE_HTTP_PROXY and TRADINGAGENTS_YFINANCE_HTTPS_PROXY.
    """
    shared_proxy = os.getenv("TRADINGAGENTS_YFINANCE_PROXY") or os.getenv("YFINANCE_PROXY")
    http_proxy = os.getenv("TRADINGAGENTS_YFINANCE_HTTP_PROXY")
    https_proxy = os.getenv("TRADINGAGENTS_YFINANCE_HTTPS_PROXY")

    if shared_proxy is not None:
        if _disabled(shared_proxy):
            return None
        return {"http": shared_proxy, "https": shared_proxy}

    proxies = {}
    if http_proxy is not None and not _disabled(http_proxy):
        proxies["http"] = http_proxy
    if https_proxy is not None and not _disabled(https_proxy):
        proxies["https"] = https_proxy
    if proxies:
        return proxies

    # Backward compatibility with the previous hard-coded local proxy.
    return {"http": LEGACY_YFINANCE_PROXY, "https": LEGACY_YFINANCE_PROXY}


def configure_yfinance_proxy(yf_module):
    """Apply the shared yfinance proxy configuration."""
    proxies = get_yfinance_proxies()
    network_config = getattr(getattr(yf_module, "config", None), "network", None)
    if network_config is not None:
        network_config.proxy = proxies
        return

    set_config = getattr(yf_module, "set_config", None)
    if callable(set_config):
        set_config(proxy=proxies)
