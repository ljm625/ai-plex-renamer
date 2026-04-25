import unittest
from pathlib import Path
from unittest.mock import patch

from ai_plex_renamer.models import MediaGuess
from ai_plex_renamer.renamer import make_rename_plan
from ai_plex_renamer.tmdb import TMDBClient


class TMDBClientTests(unittest.TestCase):
    def test_from_environment_returns_none_without_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(TMDBClient.from_environment())

    def test_enrich_movie_adds_standard_title_and_year(self):
        client = TMDBClient(bearer_token="token", transport=_transport_for_movie)

        guess = client.enrich(MediaGuess(media_type="movie", title="Inception", confidence=0.8))

        self.assertEqual(guess.title, "Inception")
        self.assertEqual(guess.year, 2010)
        self.assertIn("TMDB matched movie", guess.reason)

    def test_verbose_logs_request_and_response(self):
        logs = []
        client = TMDBClient(bearer_token="token", api_key="key", transport=_transport_for_movie, debug=logs.append)

        client.enrich(MediaGuess(media_type="movie", title="Inception", confidence=0.8))

        output = "\n".join(logs)
        self.assertIn("[verbose] tmdb request", output)
        self.assertIn("[verbose] tmdb response", output)
        self.assertNotIn("Bearer token", output)
        self.assertNotIn('"api_key": "key"', output)

    def test_query_is_nfc_normalized_and_can_include_adult(self):
        captured = {}

        def transport(url, headers, timeout):
            captured["url"] = url
            return {"results": []}

        client = TMDBClient(api_key="key", include_adult=True, transport=transport)
        decomposed_title = "キ\u3099ルティホール"

        client.enrich(MediaGuess(media_type="tv", title=decomposed_title, season=1, episode=1))

        self.assertIn("include_adult=true", captured["url"])
        self.assertIn("query=%E3%82%AE%E3%83%AB%E3%83%86%E3%82%A3%E3%83%9B%E3%83%BC%E3%83%AB", captured["url"])

    def test_enrich_tv_adds_year_and_episode_title(self):
        client = TMDBClient(bearer_token="token", transport=_transport_for_tv)

        guess = client.enrich(
            MediaGuess(media_type="tv", title="The Last of Us", season=1, episode=2, confidence=0.85)
        )

        self.assertEqual(guess.title, "The Last of Us")
        self.assertEqual(guess.year, 2023)
        self.assertEqual(guess.episode_title, "Infected")
        self.assertIn("TMDB matched TV", guess.reason)

    def test_tmdb_failure_returns_original_guess_with_reason(self):
        def failing_transport(url, headers, timeout):
            raise RuntimeError("boom")

        client = TMDBClient(bearer_token="token", transport=failing_transport)
        guess = client.enrich(MediaGuess(media_type="movie", title="Inception", year=2010))

        self.assertEqual(guess.title, "Inception")
        self.assertEqual(guess.year, 2010)
        self.assertIn("TMDB lookup failed: boom", guess.reason)


class TMDBRenameTests(unittest.TestCase):
    def test_tmdb_year_is_used_in_movie_filename(self):
        client = TMDBClient(bearer_token="token", transport=_transport_for_movie)

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="movie",
                title="Inception",
                confidence=0.55,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(
                Path("/tmp/Inception.1080p.mkv"),
                Path("/tmp"),
                classifier=None,
                tmdb_client=client,
            )

        self.assertEqual(plan.target.name, "Inception (2010).mkv")

    def test_tmdb_year_is_used_in_tv_filename(self):
        client = TMDBClient(bearer_token="token", transport=_transport_for_tv)

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="tv",
                title="The Last of Us",
                season=1,
                episode=2,
                confidence=0.85,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(
                Path("/tmp/The.Last.of.Us.S01E02.mkv"),
                Path("/tmp"),
                classifier=None,
                tmdb_client=client,
            )

        self.assertEqual(plan.target.name, "The Last of Us (2023) - S01E02 - Infected.mkv")


def _transport_for_movie(url, headers, timeout):
    assert headers["Authorization"] == "Bearer token"
    assert "search/movie" in url
    return {
        "results": [
            {
                "id": 27205,
                "title": "Inception",
                "original_title": "Inception",
                "release_date": "2010-07-15",
                "popularity": 80,
            }
        ]
    }


def _transport_for_tv(url, headers, timeout):
    assert headers["Authorization"] == "Bearer token"
    if "search/tv" in url:
        return {
            "results": [
                {
                    "id": 100088,
                    "name": "The Last of Us",
                    "original_name": "The Last of Us",
                    "first_air_date": "2023-01-15",
                    "popularity": 100,
                }
            ]
        }
    if "season/1/episode/2" in url:
        return {"name": "Infected"}
    raise AssertionError(f"unexpected URL: {url}")


if __name__ == "__main__":
    unittest.main()
