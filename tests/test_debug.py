import unittest

from ai_plex_renamer.debug import redact


class DebugTests(unittest.TestCase):
    def test_redacts_sensitive_headers_and_query(self):
        redacted = redact(
            {
                "url": "https://api.example.test/search?api_key=secret&query=Movie",
                "headers": {
                    "Authorization": "Bearer secret",
                    "Accept": "application/json",
                },
                "api_key": "secret",
            }
        )

        self.assertEqual(redacted["url"], "https://api.example.test/search?api_key=***REDACTED***&query=Movie")
        self.assertEqual(redacted["headers"]["Authorization"], "***REDACTED***")
        self.assertEqual(redacted["headers"]["Accept"], "application/json")
        self.assertEqual(redacted["api_key"], "***REDACTED***")


if __name__ == "__main__":
    unittest.main()
