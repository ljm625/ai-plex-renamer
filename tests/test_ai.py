import unittest
from pathlib import Path

from ai_plex_renamer.ai import (
    NVIDIA_DEFAULT_BASE_URL,
    NvidiaAIClassifier,
    parse_ai_batch_response,
    parse_ai_response,
)
from ai_plex_renamer.models import MediaGuess


class AIResponseTests(unittest.TestCase):
    def test_parse_ai_response_accepts_json_fence(self):
        guess = parse_ai_response(
            """```json
            {"media_type":"movie","title":"Inception","year":2010,"confidence":0.9}
            ```"""
        )

        self.assertEqual(guess.media_type, "movie")
        self.assertEqual(guess.title, "Inception")
        self.assertEqual(guess.year, 2010)

    def test_parse_ai_batch_response_maps_indexes_to_paths(self):
        paths = [Path("/tmp/show/ep1.mkv"), Path("/tmp/show/ep2.mkv")]

        guesses = parse_ai_batch_response(
            """
            {
              "files": [
                {"index": 1, "media_type": "tv", "title": "Show", "season": 1, "episode": 1},
                {"index": 2, "media_type": "tv", "title": "Show", "season": 1, "episode": 2}
              ]
            }
            """,
            paths,
        )

        self.assertEqual(guesses[paths[0]].episode, 1)
        self.assertEqual(guesses[paths[1]].episode, 2)

    def test_parse_ai_response_requires_object(self):
        with self.assertRaises(ValueError):
            parse_ai_response("[]")

    def test_nvidia_classifier_calls_chat_completions(self):
        captured = {}

        def transport(url, headers, payload, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"media_type":"movie","title":"Inception","year":2010,"confidence":0.9}'
                        }
                    }
                ]
            }

        classifier = NvidiaAIClassifier(
            api_key="nvapi-test",
            model="meta/test-model",
            timeout=12,
            transport=transport,
        )

        guess = classifier.classify(
            Path("/tmp/Inception.1080p.mkv"),
            Path("/tmp"),
            "auto",
            MediaGuess(media_type="movie", title="Inception", confidence=0.55),
        )

        self.assertEqual(captured["url"], f"{NVIDIA_DEFAULT_BASE_URL}/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer nvapi-test")
        self.assertEqual(captured["payload"]["model"], "meta/test-model")
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(guess.title, "Inception")
        self.assertEqual(guess.year, 2010)

    def test_nvidia_classifier_verbose_logs_request_and_response(self):
        logs = []

        def transport(url, headers, payload, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"media_type":"movie","title":"Inception","year":2010,"confidence":0.9}'
                        }
                    }
                ]
            }

        classifier = NvidiaAIClassifier(
            api_key="nvapi-test",
            transport=transport,
            debug=logs.append,
        )
        classifier.classify(
            Path("/tmp/Inception.1080p.mkv"),
            Path("/tmp"),
            "auto",
            MediaGuess(media_type="movie", title="Inception", confidence=0.55),
        )

        output = "\n".join(logs)
        self.assertIn("[verbose] nvidia request", output)
        self.assertIn("[verbose] nvidia response", output)
        self.assertNotIn("nvapi-test", output)


if __name__ == "__main__":
    unittest.main()
