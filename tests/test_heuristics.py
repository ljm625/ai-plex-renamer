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
