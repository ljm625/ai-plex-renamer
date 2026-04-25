from __future__ import annotations

import urllib.request
from typing import Any


PROXY_SCHEMES = ("http", "https")


def urlopen_with_environment_proxy(request: urllib.request.Request, timeout: int) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler())
    return opener.open(request, timeout=timeout)


def proxy_debug_info() -> dict[str, str]:
    proxies = urllib.request.getproxies()
    info = {
        scheme: "configured"
        for scheme in PROXY_SCHEMES
        if proxies.get(scheme)
    }
    if proxies.get("no"):
        info["no_proxy"] = "configured"
    return info
