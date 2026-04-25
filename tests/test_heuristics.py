import unittest
from pathlib import Path

from ai_plex_renamer.heuristics import guess_from_filename


class HeuristicTests(unittest.TestCase):
    def test_guess_tv_sxxexx(self):
        guess = guess_from_filename(Path("The.Last.of.Us.S01E02.Infected.1080p.WEB-DL.mkv"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "The Last of Us")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 2)
        self.assertEqual(guess.episode_title, "Infected")

    def test_guess_tv_x_pattern(self):
        guess = guess_from_filename(Path("Breaking.Bad.2x03.Bit.by.a.Dead.Bee.mkv"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "Breaking Bad")
        self.assertEqual(guess.season, 2)
        self.assertEqual(guess.episode, 3)
        self.assertEqual(guess.episode_title, "Bit by a Dead Bee")

    def test_guess_movie_with_year(self):
        guess = guess_from_filename(Path("Inception.2010.1080p.BluRay.mkv"))

        self.assertEqual(guess.media_type, "movie")
        self.assertEqual(guess.title, "Inception")
        self.assertEqual(guess.year, 2010)

    def test_guess_folder_episode_with_japanese_part_marker(self):
        guess = guess_from_filename(
            Path("/tmp/test_folder/服はまで/[260130][Louis]服はまで 前編[り].chs.mp4")
        )

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "服はまで")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 1)
        self.assertEqual(guess.episode_title, "前編")

    def test_guess_folder_episode_with_fullwidth_hash_marker(self):
        guess = guess_from_filename(Path("/tmp/OVA Louis/Louis ＃2.mp4"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "Louis")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 2)
        self.assertIsNone(guess.episode_title)

    def test_guess_folder_episode_with_bracketed_release_number(self):
        guess = guess_from_filename(
            Path("/tmp/Overflow/[Sakurato.sub][Overflow][01][GB][HEVC-10Bit][1080P].mp4")
        )

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "Overflow")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 1)
        self.assertIsNone(guess.episode_title)

    def test_guess_folder_episode_with_title_dash_number_crc(self):
        guess = guess_from_filename(Path("/tmp/Overflow/Overflow - 01 (0ED62047).mkv"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "Overflow")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 1)
        self.assertIsNone(guess.episode_title)

    def test_guess_japanese_volume_marker_uses_filename_title_and_ignores_version_suffix(self):
        guess = guess_from_filename(Path("/tmp/Library/作品名 THE ANIMATION 第2巻 v2.mkv"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "作品名 THE ANIMATION")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 2)
        self.assertIsNone(guess.episode_title)

    def test_guess_japanese_kanji_episode_marker(self):
        guess = guess_from_filename(Path("作品名 第一話.mkv"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "作品名")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 1)
        self.assertIsNone(guess.episode_title)

    def test_guess_japanese_kanji_episode_marker_above_ten(self):
        guess = guess_from_filename(Path("作品名 第十二話.mkv"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "作品名")
        self.assertEqual(guess.episode, 12)

    def test_guess_folder_episode_prefers_bracket_title_over_noisy_parent(self):
        guess = guess_from_filename(
            Path(
                "/tmp/[KyokuSai] Harem Camp! [01-08][720P][WEB-DL][UNC]/"
                "[Eternal][Harem Camp!][01][GB][720P][Premium].mp4"
            )
        )

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "Harem Camp!")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 1)
        self.assertIsNone(guess.episode_title)

    def test_guess_special_cm_uses_season_zero_and_grandparent_title(self):
        guess = guess_from_filename(
            Path(
                "/tmp/[Nekomoe kissaten&VCB-Studio] Araiya-san! Ore to Aitsu ga Onnayu de! [Ma10p_1080p]/SPs/"
                "[Nekomoe kissaten&VCB-Studio] Araiya-san! Ore to Aitsu ga Onnayu de! [CM04][Ma10p_1080p][x265_flac].mkv"
            )
        )

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "Araiya-san! Ore to Aitsu ga Onnayu de!")
        self.assertEqual(guess.season, 0)
        self.assertEqual(guess.episode, 4)
        self.assertEqual(guess.episode_title, "CM04")

    def test_guess_unnumbered_special_is_not_usable(self):
        guess = guess_from_filename(
            Path(
                "/tmp/Araiya-san! Ore to Aitsu ga Onnayu de!/SPs/"
                "[Nekomoe kissaten&VCB-Studio] Araiya-san! Ore to Aitsu ga Onnayu de! [Menu][Ma10p_1080p][x265].mkv"
            )
        )

        self.assertEqual(guess.media_type, "unknown")
        self.assertIn("has no episode number", guess.reason)

    def test_sxxexx_without_title_uses_parent_folder(self):
        guess = guess_from_filename(Path("/tmp/LouisTheCat/S1E03.mp4"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "LouisTheCat")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 3)

    def test_sxxexx_language_tag_suffix_uses_parent_folder_without_episode_title(self):
        guess = guess_from_filename(Path("/tmp/LouisTheCat/S01E02.chs.mp4"))

        self.assertEqual(guess.media_type, "tv")
        self.assertEqual(guess.title, "LouisTheCat")
        self.assertEqual(guess.season, 1)
        self.assertEqual(guess.episode, 2)
        self.assertIsNone(guess.episode_title)


if __name__ == "__main__":
    unittest.main()
