import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from ai_plex_renamer.guessit_parser import media_guess_from_guessit
from ai_plex_renamer.models import MediaGuess, RenamePlan
from ai_plex_renamer.renamer import apply_plan, apply_plans_by_group, build_plans, make_rename_plan, validate_apply_plans


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

    def test_title_dash_number_files_are_episode_planned_without_ai(self):
        files = [
            Path("/tmp/Hentai2/Overflow/Overflow - 01 (0ED62047).mkv"),
            Path("/tmp/Hentai2/Overflow/Overflow - 02 (7841C3C7).mkv"),
        ]

        plans = build_plans(files, Path("/tmp/Hentai2"), classifier=None, tmdb_client=None, library_hint="auto", collision="skip")

        self.assertEqual(plans[0].target.name, "Overflow - S01E01.mkv")
        self.assertEqual(plans[1].target.name, "Overflow - S01E02.mkv")

    def test_japanese_volume_marker_file_uses_filename_title_not_parent_folder(self):
        source = Path("/tmp/Library/作品名 THE ANIMATION 第2巻 v2.mkv")

        plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None, tmdb_client=None)

        self.assertEqual(plan.target, Path("/tmp/Library/作品名 THE ANIMATION/作品名 THE ANIMATION - S01E02.mkv"))
        self.assertEqual(plan.guess.title, "作品名 THE ANIMATION")
        self.assertEqual(plan.guess.episode, 2)

    def test_japanese_kanji_episode_marker_file_is_planned(self):
        source = Path("/tmp/Library/作品名 第一話.mkv")

        plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None, tmdb_client=None)

        self.assertEqual(plan.target, Path("/tmp/Library/作品名/作品名 - S01E01.mkv"))
        self.assertEqual(plan.guess.episode, 1)

    def test_bracket_title_beats_noisy_parent_directory(self):
        parent = Path("/tmp/Hentai2/[KyokuSai] Harem Camp! [01-08][720P][WEB-DL][UNC]")
        files = [
            parent / "[Eternal][Harem Camp!][01][GB][720P][Premium].mp4",
            parent / "[Eternal][Harem Camp!][02][GB][720P][Premium].mp4",
        ]

        plans = build_plans(files, Path("/tmp/Hentai2"), classifier=None, tmdb_client=None, library_hint="auto", collision="skip")

        self.assertEqual(plans[0].target.name, "Harem Camp! - S01E01.mp4")
        self.assertEqual(plans[1].target.name, "Harem Camp! - S01E02.mp4")

    def test_special_cm_files_use_season_zero_and_specials_folder(self):
        parent = Path("/tmp/Hentai2/[Nekomoe kissaten&VCB-Studio] Araiya-san! Ore to Aitsu ga Onnayu de! [Ma10p_1080p]/SPs")
        files = [
            parent / "[Nekomoe kissaten&VCB-Studio] Araiya-san! Ore to Aitsu ga Onnayu de! [CM04][Ma10p_1080p][x265_flac].mkv",
            parent / "[Nekomoe kissaten&VCB-Studio] Araiya-san! Ore to Aitsu ga Onnayu de! [CM05][Ma10p_1080p][x265_flac].mkv",
        ]

        plans = build_plans(files, Path("/tmp/Hentai2"), classifier=None, tmdb_client=None, library_hint="auto", collision="skip")

        self.assertEqual(plans[0].target.parent.name, "Specials")
        self.assertEqual(plans[0].target.name, "Araiya-san! Ore to Aitsu ga Onnayu de! - S00E04 - CM04.mkv")
        self.assertEqual(plans[1].target.name, "Araiya-san! Ore to Aitsu ga Onnayu de! - S00E05 - CM05.mkv")

    def test_unnumbered_special_files_skip_ai(self):
        source = Path(
            "/tmp/Hentai2/Araiya-san! Ore to Aitsu ga Onnayu de!/SPs/"
            "[Nekomoe kissaten&VCB-Studio] Araiya-san! Ore to Aitsu ga Onnayu de! [Menu][Ma10p_1080p][x265].mkv"
        )
        classifier = _DummyClassifier(MediaGuess(media_type="tv", title="Wrong", season=1, episode=7))

        plan = make_rename_plan(source, Path("/tmp/Hentai2"), classifier=classifier)

        self.assertEqual(classifier.calls, 0)
        self.assertEqual(plan.status, "skipped")
        self.assertIn("has no episode number", plan.message)

    def test_type_tv_single_unparsed_file_defaults_to_s01e01_from_filename(self):
        source = Path("/tmp/Library/Lonely Show 1080p.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess.unknown("GuessIt did not find title, season, and episode."),
        ):
            plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None, library_hint="tv")

        self.assertEqual(plan.target, Path("/tmp/Library/Lonely Show/Lonely Show - S01E01.mkv"))
        self.assertEqual(plan.guess.media_type, "tv")
        self.assertEqual(plan.guess.title, "Lonely Show")
        self.assertEqual(plan.guess.season, 1)
        self.assertEqual(plan.guess.episode, 1)

    def test_type_tv_single_unparsed_file_uses_parent_title_inside_show_folder(self):
        source = Path("/tmp/Library/Lonely Show/weird-release-name.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess.unknown("GuessIt did not find title, season, and episode."),
        ):
            plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None, library_hint="tv")

        self.assertEqual(plan.target, Path("/tmp/Library/Lonely Show/Lonely Show - S01E01.mkv"))
        self.assertEqual(plan.guess.title, "Lonely Show")

    def test_type_tv_multiple_unparsed_root_files_default_when_titles_are_unique(self):
        files = [
            Path("/tmp/Library/Lonely Show 1080p.mkv"),
            Path("/tmp/Library/Another Special.mkv"),
        ]

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess.unknown("GuessIt did not find title, season, and episode."),
        ):
            plans = build_plans(files, Path("/tmp/Library"), classifier=None, tmdb_client=None, library_hint="tv", collision="skip")

        self.assertEqual(plans[0].target, Path("/tmp/Library/Lonely Show/Lonely Show - S01E01.mkv"))
        self.assertEqual(plans[1].target, Path("/tmp/Library/Another Special/Another Special - S01E01.mkv"))

    def test_type_tv_multiple_unparsed_files_do_not_default_to_s01e01(self):
        files = [
            Path("/tmp/Library/Lonely Show/weird-a.mkv"),
            Path("/tmp/Library/Lonely Show/weird-b.mkv"),
        ]

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess.unknown("GuessIt did not find title, season, and episode."),
        ):
            plans = build_plans(files, Path("/tmp/Library"), classifier=None, tmdb_client=None, library_hint="tv", collision="skip")

        self.assertEqual(plans[0].status, "skipped")
        self.assertEqual(plans[1].status, "skipped")

    def test_type_tv_multiple_unparsed_root_files_with_same_title_do_not_default(self):
        files = [
            Path("/tmp/Library/Lonely Show - 01.mkv"),
            Path("/tmp/Library/Lonely Show - 02.mkv"),
        ]

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess.unknown("GuessIt did not find title, season, and episode."),
        ):
            plans = build_plans(files, Path("/tmp/Library"), classifier=None, tmdb_client=None, library_hint="tv", collision="skip")

        self.assertEqual(plans[0].status, "skipped")
        self.assertEqual(plans[1].status, "skipped")

    def test_type_tv_single_unparsed_numeric_filename_does_not_default(self):
        source = Path("/tmp/Library/01.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess.unknown("GuessIt did not find title, season, and episode."),
        ):
            plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None, library_hint="tv")

        self.assertEqual(plan.status, "skipped")

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

    def test_root_tv_episode_with_matching_filename_still_moves_into_show_folder(self):
        source = Path("/tmp/Library/Treme - S01E03.mkv")

        with patch(
            "ai_plex_renamer.renamer.guess_with_guessit",
            return_value=MediaGuess(
                media_type="tv",
                title="Treme",
                season=1,
                episode=3,
                confidence=0.85,
                reason="Parsed by GuessIt.",
            ),
        ):
            plan = make_rename_plan(source, Path("/tmp/Library"), classifier=None)

        self.assertEqual(plan.status, "planned")
        self.assertEqual(plan.target, Path("/tmp/Library/Treme/Treme - S01E03.mkv"))

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

    def test_apply_plan_reports_missing_source_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "missing.mkv"
            target = root / "target.mkv"
            plan = RenamePlan(
                source=source,
                target=target,
                guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                status="planned",
            )

            applied = apply_plan(plan, retry_delays=())

            self.assertEqual(applied.status, "error")
            self.assertIn("Source file was not found at apply time.", applied.message)
            self.assertIn("source_exists=False", applied.message)
            self.assertIn("target_parent_exists=True", applied.message)

    def test_apply_plan_retries_transient_missing_source_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Show.1x01.mkv"
            target = root / "Show - S01E01.mkv"
            source.touch()
            plan = RenamePlan(
                source=source,
                target=target,
                guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                status="planned",
            )
            calls = []
            real_rename = Path.rename

            def flaky_rename(self, destination):
                if self == source and not calls:
                    calls.append("failed")
                    raise FileNotFoundError(2, "simulated transient missing source", str(self))
                return real_rename(self, destination)

            with patch("pathlib.Path.rename", flaky_rename):
                applied = apply_plan(plan, retry_delays=(0,))

            self.assertEqual(applied.status, "renamed")
            self.assertTrue(target.exists())

    def test_apply_plan_uses_copy_fallback_when_rename_cannot_find_existing_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Show.1x01.mkv"
            target = root / "Show - S01E01.mkv"
            source.write_text("video", encoding="utf-8")
            plan = RenamePlan(
                source=source,
                target=target,
                guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                status="planned",
            )

            def failing_rename(self, destination):
                raise FileNotFoundError(2, "simulated persistent missing source", str(self))

            with patch("pathlib.Path.rename", failing_rename):
                applied = apply_plan(plan, retry_delays=())

            self.assertEqual(applied.status, "renamed")
            self.assertEqual(applied.message, "Renamed successfully via copy fallback.")
            self.assertFalse(source.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "video")

    def test_apply_plan_moves_matching_subtitle_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Show.1x01.mkv"
            target = root / "Show - S01E01.mkv"
            source.write_text("video", encoding="utf-8")
            (root / "Show.1x01.ass").write_text("ass", encoding="utf-8")
            (root / "Show.1x01.SC.ass").write_text("sc", encoding="utf-8")
            (root / "Show.1x01.TC.ass").write_text("tc", encoding="utf-8")
            (root / "Show.1x01.chi.srt").write_text("chi", encoding="utf-8")
            unrelated = root / "Other.ass"
            unrelated.write_text("other", encoding="utf-8")
            plan = RenamePlan(
                source=source,
                target=target,
                guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                status="planned",
            )

            applied = apply_plan(plan, retry_delays=())

            self.assertEqual(applied.status, "renamed")
            self.assertFalse((root / "Show.1x01.ass").exists())
            self.assertEqual((root / "Show - S01E01.ass").read_text(encoding="utf-8"), "ass")
            self.assertEqual((root / "Show - S01E01.zh-Hans.ass").read_text(encoding="utf-8"), "sc")
            self.assertEqual((root / "Show - S01E01.zh-Hant.ass").read_text(encoding="utf-8"), "tc")
            self.assertEqual((root / "Show - S01E01.chi.srt").read_text(encoding="utf-8"), "chi")
            self.assertTrue(unrelated.exists())
            self.assertIn("Moved 4 sidecar", applied.message)

    def test_apply_plan_rolls_back_video_when_sidecar_move_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Show.1x01.mkv"
            sidecar = root / "Show.1x01.ass"
            target = root / "Show - S01E01.mkv"
            source.write_text("video", encoding="utf-8")
            sidecar.write_text("ass", encoding="utf-8")
            plan = RenamePlan(
                source=source,
                target=target,
                guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                status="planned",
            )
            real_rename = Path.rename

            def failing_sidecar_rename(self, destination):
                if self == sidecar:
                    raise OSError("simulated sidecar move failure")
                return real_rename(self, destination)

            with patch("pathlib.Path.rename", failing_sidecar_rename):
                applied = apply_plan(plan, retry_delays=())

            self.assertEqual(applied.status, "error")
            self.assertIn("Rolled back moved file", applied.message)
            self.assertTrue(source.exists())
            self.assertTrue(sidecar.exists())
            self.assertFalse(target.exists())

    def test_validate_apply_plans_reports_duplicate_targets_before_renaming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.mkv"
            second = root / "second.mkv"
            target = root / "Show - S01E01.mkv"
            first.touch()
            second.touch()
            guess = MediaGuess(media_type="tv", title="Show", season=1, episode=1)
            plans = [
                RenamePlan(source=first, target=target, guess=guess, status="planned"),
                RenamePlan(source=second, target=target, guess=guess, status="planned"),
            ]

            problems = validate_apply_plans(plans)

            self.assertEqual(len(problems), 1)
            self.assertIn("Duplicate target planned", problems[0])

    def test_apply_plans_by_group_moves_duplicate_target_loser_to_failed_with_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = root / "Adam's Sweet Agony"
            parent.mkdir()
            first = parent / "[KyokuSai] Adam's Sweet Agony [03][720P].mp4"
            second = parent / "[KyokuSai] Adam's Sweet Agony [03.5][720P].mp4"
            target = parent / "Adam's Sweet Agony (2024) - S01E03 - I Stopped Hiding It.mp4"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            sidecar = parent / "[KyokuSai] Adam's Sweet Agony [03.5][720P].SC.ass"
            sidecar.write_text("subtitle", encoding="utf-8")
            guess = MediaGuess(media_type="tv", title="Adam's Sweet Agony", year=2024, season=1, episode=3)
            plans = [
                RenamePlan(source=first, target=target, guess=guess, status="planned"),
                RenamePlan(source=second, target=target, guess=guess, status="planned"),
            ]

            applied = apply_plans_by_group(plans, root, retry_delays=())

            self.assertEqual(applied[0].status, "renamed")
            self.assertEqual(applied[1].status, "failed")
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "first")
            failed_video = root / ".failed" / "Adam's Sweet Agony" / second.name
            failed_sidecar = root / ".failed" / "Adam's Sweet Agony" / sidecar.name
            self.assertTrue(failed_video.exists())
            self.assertTrue(failed_sidecar.exists())
            self.assertFalse(second.exists())
            self.assertFalse(sidecar.exists())

    def test_apply_plans_by_group_keeps_other_folders_running_after_group_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            broken_parent = root / "Broken"
            good_parent = root / "Good"
            broken_parent.mkdir()
            good_parent.mkdir()
            broken_source = broken_parent / "Show.1x01.mkv"
            blocked_parent = broken_parent / "Blocked"
            broken_target = blocked_parent / "Show - S01E01.mkv"
            good_source = good_parent / "Good.1x01.mkv"
            good_target = good_parent / "Good - S01E01.mkv"
            broken_source.write_text("broken", encoding="utf-8")
            blocked_parent.write_text("not a folder", encoding="utf-8")
            good_source.write_text("good", encoding="utf-8")
            plans = [
                RenamePlan(
                    source=broken_source,
                    target=broken_target,
                    guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                    status="planned",
                ),
                RenamePlan(
                    source=good_source,
                    target=good_target,
                    guess=MediaGuess(media_type="tv", title="Good", season=1, episode=1),
                    status="planned",
                ),
            ]

            applied = apply_plans_by_group(plans, root, retry_delays=())

            self.assertEqual(applied[0].status, "error")
            self.assertEqual(applied[1].status, "renamed")
            self.assertTrue(broken_source.exists())
            self.assertTrue(blocked_parent.is_file())
            self.assertTrue(good_target.exists())
            self.assertFalse(good_source.exists())

    def test_apply_plans_by_group_moves_existing_target_conflict_to_failed_and_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = root / "Show"
            parent.mkdir()
            source = parent / "Show.1x01.alt.mkv"
            target = parent / "Show - S01E01.mkv"
            next_source = parent / "Show.1x02.mkv"
            next_target = parent / "Show - S01E02.mkv"
            source.write_text("alternate", encoding="utf-8")
            target.write_text("existing", encoding="utf-8")
            next_source.write_text("next", encoding="utf-8")
            plans = [
                RenamePlan(
                    source=source,
                    target=target,
                    guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                    status="planned",
                ),
                RenamePlan(
                    source=next_source,
                    target=next_target,
                    guess=MediaGuess(media_type="tv", title="Show", season=1, episode=2),
                    status="planned",
                ),
            ]

            applied = apply_plans_by_group(plans, root, retry_delays=())

            self.assertEqual(applied[0].status, "failed")
            self.assertEqual(applied[1].status, "renamed")
            self.assertEqual(target.read_text(encoding="utf-8"), "existing")
            self.assertTrue((root / ".failed" / "Show" / source.name).exists())
            self.assertFalse(source.exists())
            self.assertTrue(next_target.exists())
            self.assertFalse(next_source.exists())

    def test_apply_plan_blocks_duplicate_normalized_sidecar_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Show.1x01.mkv"
            target = root / "Show" / "Show - S01E01.mkv"
            source.write_text("video", encoding="utf-8")
            (root / "Show.1x01.SC.ass").write_text("sc", encoding="utf-8")
            (root / "Show.1x01.chs.ass").write_text("chs", encoding="utf-8")
            plan = RenamePlan(
                source=source,
                target=target,
                guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                status="planned",
            )

            applied = apply_plan(plan, retry_delays=())

            self.assertEqual(applied.status, "error")
            self.assertIn("same target filename", applied.message)
            self.assertTrue(source.exists())
            self.assertFalse(target.exists())
            self.assertFalse(target.parent.exists())

    def test_validate_apply_plans_reports_duplicate_normalized_sidecar_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "Show.1x01.mkv"
            target = root / "Show - S01E01.mkv"
            source.touch()
            (root / "Show.1x01.SC.ass").touch()
            (root / "Show.1x01.chs.ass").touch()
            plan = RenamePlan(
                source=source,
                target=target,
                guess=MediaGuess(media_type="tv", title="Show", season=1, episode=1),
                status="planned",
            )

            problems = validate_apply_plans([plan])

            self.assertEqual(len(problems), 1)
            self.assertIn("Duplicate target planned", problems[0])

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
