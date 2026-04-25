import unittest
import urllib.request
from unittest.mock import patch

from ai_plex_renamer.http_client import proxy_debug_info, urlopen_with_environment_proxy


class HTTPClientTests(unittest.TestCase):
    def test_urlopen_uses_environment_proxy_handler(self):
        class DummyOpener:
            def __init__(self):
                self.request = None
                self.timeout = None

            def open(self, request, timeout):
                self.request = request
                self.timeout = timeout
                return "response"

        opener = DummyOpener()
        request = urllib.request.Request("https://example.com")

        with patch("urllib.request.build_opener", return_value=opener) as build_opener:
            response = urlopen_with_environment_proxy(request, timeout=12)

        self.assertEqual(response, "response")
        self.assertIs(opener.request, request)
        self.assertEqual(opener.timeout, 12)
        self.assertIsInstance(build_opener.call_args.args[0], urllib.request.ProxyHandler)

    def test_proxy_debug_info_reports_configured_proxies_without_credentials(self):
        with patch(
            "urllib.request.getproxies",
            return_value={
                "http": "http://proxy.local:8080",
                "https": "http://user:secret@proxy.local:8080",
                "no": "localhost,127.0.0.1",
            },
        ):
            info = proxy_debug_info()

        self.assertEqual(info, {"http": "configured", "https": "configured", "no_proxy": "configured"})
        self.assertNotIn("secret", str(info))


if __name__ == "__main__":
    unittest.main()
