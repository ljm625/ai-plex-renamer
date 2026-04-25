import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from ai_plex_renamer.guessit_parser import media_guess_from_guessit
from ai_plex_renamer.models import MediaGuess
from ai_plex_renamer.renamer import apply_plan, build_plans, make_rename_plan


class GuessItParserTests(unittest.TestCase):
    def test_episode_from_guessit_three_x_fields(self):
        guess = media_guess_from_guessit(
            {
                "type": "episode",
                "title": "Treme",
                "season": 1,
                "episode": 3,
                "episode_title": "Right Place, Wrong Time",
            }
        )

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "Treme")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 3)
        self.assertEqual(guess.episode_title, "Right Place, Wrong Time")
        self.assertEqual(guess.reason, "Parsed by GuessIt.")

    def test_episode_from_guessit_old_series_fields(self):
        guess = media_guess_from_guessit(
            {
                "type": "episode",
                "series": "Treme",
                "seasonNumber": 1,
                "episodeNumber": 3,
                "title": "Right Place, Wrong Time",
            }
        )

        self.assertEqual(guess.title, "Treme")
        self.assertEqual(guess.episode_title, "Right Place, Wrong Time")

    def test_movie_from_guessit_keeps_title_without_year_for_tmdb(self):
        guess = media_guess_from_guessit({"type": "movie", "title": "Inception"})

        self.assertEqual(guess.media_type, "movie")
        self.assertEqual(guess.title, "Inception")
        self.assertIsNone(guess.year)

    def test_movie_from_guessit_accepts_title_only_with_movie_hint(self):
        guess = media_guess_from_guessit({"type": "movie", "title": "Inception"}, library_hint="movie")

        self.assertEqual(guess.media_type, "movie")
        self.assertEqual(guess.title, "Inception")
        self.assertIsNone(guess.year)

    def test_guessit_language_tag_title_is_not_usable(self):
        guess = media_guess_from_guessit({"type": "movie", "title": "chs"})

        self.assertEqual(guess.media_type, "unknown")


class RenamePriorityTests(unittest.TestCase):
    def test_folder_episode_marker_beats_guessit_movie(self):
        classifier = _DummyClassifier(MediaGuess(media_type="movie", title="Wrong", year=2000))
        source = Path("/tmp/test_folder/制服は着たままで/[260130][Queen Bee]制服は着たままで 前編[りぶつ].chs.mp4")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="movie",
                title="制服は着たままで 前編",
                confidence=0.55,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp/test_folder"), classifier)

        self.assertEqual(classifier.calls, 0)
        self.assertEqual(plan.target.name, "制服は着たままで - S01E01 - 前編.mp4")

    def test_guessit_success_skips_ai(self):
        classifier = _DummyClassifier(MediaGuess(media_type="movie", title="Wrong", year=2000))
        source = Path("/tmp/Treme.1x03.Right.Place.Wrong.Time.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="tv",
                title="Treme",
                season=1,
                episode=3,
                episode_title="Right Place Wrong Time",
                confidence=0.85,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp"), classifier)

        self.assertEqual(classifier.calls, 0)
        self.assertEqual(plan.target.name, "Treme - S01E03 - Right Place Wrong Time.mkv")

    def test_ai_is_used_when_guessit_is_unusable(self):
        classifier = _DummyClassifier(MediaGuess(media_type="movie", title="Inception", year=2010))
        source = Path("/tmp/no-clear-pattern.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess.unknown("GuessIt did not return usable Plex metadata."),
        ):
            plan = make_rename_plan(source, Path("/tmp"), classifier)

        self.assertEqual(classifier.calls, 1)
        self.assertEqual(plan.target.name, "Inception (2010).mkv")

    def test_low_confidence_movie_without_year_asks_ai(self):
        classifier = _DummyClassifier(MediaGuess(media_type="movie", title="Inception", year=2010))
        source = Path("/tmp/Inception.1080p.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="movie",
                title="Inception",
                confidence=0.55,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp"), classifier)

        self.assertEqual(classifier.calls, 1)
        self.assertEqual(plan.target.name, "Inception (2010).mkv")

    def test_directory_hash_files_are_episode_planned_without_ai(self):
        files = [
            Path("/tmp/Hentai/OVA 初恋時間。/初恋時間。 ＃1.mkv"),
            Path("/tmp/Hentai/OVA 初恋時間。/初恋時間。 ＃2.mp4"),
        ]

        plans = build_plans(files, Path("/tmp/Hentai"), classifier=None, tmdb_client=None, library_hint="auto", collision="skip")

        self.assertEqual(plans[0].target.name, "初恋時間。 - S01E01.mkv")
        self.assertEqual(plans[1].target.name, "初恋時間。 - S01E02.mp4")

    def test_bracketed_release_numbers_are_episode_planned_without_ai(self):
        files = [
            Path("/tmp/Overflow/[Sakurato.sub][Overflow][01][GB][HEVC-10Bit][1080P].mp4"),
            Path("/tmp/Overflow/[Sakurato.sub][Overflow][02][GB][HEVC-10Bit][1080P].mp4"),
        ]

        plans = build_plans(files, Path("/tmp/Hentai2"), classifier=None, tmdb_client=None, library_hint="auto", collision="skip")

        self.assertEqual(plans[0].target.name, "Overflow - S01E01.mp4")
        self.assertEqual(plans[1].target.name, "Overflow - S01E02.mp4")

    def test_root_tv_episode_moves_into_show_folder(self):
        source = Path("/tmp/Library/Treme.1x03.Right.Place.Wrong.Time.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="tv",
                title="Treme",
                season=1,
                episode=3,
                episode_title="Right Place Wrong Time",
                confidence=0.85,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None)

        self.assertEqual(plan.target, Path("/tmp/Library/Treme/Treme - S01E03 - Right Place Wrong Time.mkv"))

    def test_show_directory_tv_episode_does_not_nest_show_folder(self):
        source = Path("/tmp/Library/Treme/Treme.1x03.Right.Place.Wrong.Time.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="tv",
                title="Treme",
                season=1,
                episode=3,
                episode_title="Right Place Wrong Time",
                confidence=0.85,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None)

        self.assertEqual(plan.target, Path("/tmp/Library/Treme/Treme - S01E03 - Right Place Wrong Time.mkv"))

    def test_scan_show_directory_does_not_nest_show_folder(self):
        source = Path("/tmp/Library/Treme/Treme.1x03.Right.Place.Wrong.Time.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="tv",
                title="Treme",
                season=1,
                episode=3,
                episode_title="Right Place Wrong Time",
                confidence=0.85,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp/Library/Treme"), classifier=None)

        self.assertEqual(plan.target, Path("/tmp/Library/Treme/Treme - S01E03 - Right Place Wrong Time.mkv"))

    def test_apply_plan_creates_root_show_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Treme.1x03.Right.Place.Wrong.Time.mkv"
            source.touch()

            with patch(
                "ai_plex_renamer.renamer.guess_with_guessit",
                return_value=MediaGuess(
                    media_type="tv",
                    title="Treme",
                    season=1,
                    episode=3,
                    episode_title="Right Place Wrong Time",
                    confidence=0.85,
                    reason="Parsed by GuessIt.",
                ),
            ):
                plan = make_rename_plan(source, root, classifier=None)

            applied = apply_plan(plan)

            self.assertEqual(applied.status, "renamed")
            self.assertTrue((root / "Treme" / "Treme - S01E03 - Right Place Wrong Time.mkv").exists())

    def test_sxxexx_language_suffix_does_not_search_as_title(self):
        source = Path("/tmp/Hentai2/聖痕のアリア/S01E01.chs.mp4")
        tmdb_client = _UnexpectedTMDBClient()

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="movie",
                title="chs",
                confidence=0.55,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp/Hentai2"), classifier=None, tmdb_client=tmdb_client)

        self.assertEqual(plan.target.name, "聖痕のアリア - S01E01.mp4")
        self.assertEqual(tmdb_client.queries, ["聖痕のアリア"])

    def test_directory_ai_fallback_is_batched(self):
        files = [
            Path("/tmp/Some Folder/weird-a.mkv"),
            Path("/tmp/Some Folder/weird-b.mp4"),
        ]
        classifier = _BatchDummyClassifier(
            {
                files[0]: MediaGuess(media_type="tv", title="Some Folder", season=1, episode=1),
                files[1]: MediaGuess(media_type="tv", title="Some Folder", season=1, episode=2),
            }
        )

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="movie",
                title="weird",
                confidence=0.55,
                reason="Parsed by GuessIt.",
            ),
        ):
            plans = build_plans(files, Path("/tmp"), classifier, tmdb_client=None, library_hint="auto", collision="skip")

        self.assertEqual(classifier.calls, 1)
        self.assertEqual(classifier.last_files, files)
        self.assertEqual(plans[0].target.name, "Some Folder - S01E01.mkv")
        self.assertEqual(plans[1].target.name, "Some Folder - S01E02.mp4")


class _DummyClassifier:
    def __init__(self, guess):
        self.guess = guess
        self.calls = 0

    def classify(self, source, root, library_hint, local_guess):
        self.calls += 1
        return self.guess


class _BatchDummyClassifier:
    def __init__(self, guesses):
        self.guesses = guesses
        self.calls = 0
        self.last_files = None

    def classify_many(self, files, root, library_hint, local_guesses):
        self.calls += 1
        self.last_files = files
        return self.guesses


class _UnexpectedTMDBClient:
    def __init__(self):
        self.queries = []

    def enrich(self, guess):
        self.queries.append(guess.title)
        self.assert_not_chs(guess)
        return guess

    def assert_not_chs(self, guess):
        if guess.title.lower() == "chs":
            raise AssertionError("TMDB should not be queried with a language tag title.")


if __name__ == "__main__":
    unittest.main()
