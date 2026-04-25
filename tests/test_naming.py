import unittest
from pathlib import Path

from ai_plex_renamer.models import MediaGuess
from ai_plex_renamer.naming import build_plex_filename, resolve_collision, sanitize_component


class NamingTests(unittest.TestCase):
    def test_build_tv_filename(self):
        filename = build_plex_filename(
            MediaGuess(
                media_type="tv",
                title="The Last of Us",
                season=1,
                episode=2,
                episode_title="Infected",
            ),
            ".mkv",
        )

        self.assertEqual(filename, "The Last of Us - S01E02 - Infected.mkv")

    def test_build_tv_filename_includes_year_when_available(self):
        filename = build_plex_filename(
            MediaGuess(media_type="tv", title="Band of Brothers", year=2001, season=1, episode=1),
            ".mkv",
        )

        self.assertEqual(filename, "Band of Brothers (2001) - S01E01.mkv")

    def test_build_movie_filename(self):
        filename = build_plex_filename(
            MediaGuess(media_type="movie", title="Inception", year=2010),
            ".mp4",
        )

        self.assertEqual(filename, "Inception (2010).mp4")

    def test_sanitize_component_removes_filesystem_characters(self):
        self.assertEqual(sanitize_component('Bad:/Name*?"'), "Bad Name")

    def test_resolve_collision_suffix(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            source = tmp_path / "old.mkv"
            target = tmp_path / "new.mkv"
            source.touch()
            target.touch()

            self.assertEqual(resolve_collision(target, source, "suffix"), tmp_path / "new (1).mkv")


if __name__ == "__main__":
    unittest.main()
