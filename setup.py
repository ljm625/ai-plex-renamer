from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


setup(
    name="ai-plex-renamer",
    version="0.1.0",
    description="Rename movies and TV episodes to Plex-friendly file names with GuessIt, TMDB, and NVIDIA AI fallback.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=[
        "guessit>=3.8.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-plex-renamer=ai_plex_renamer.cli:main",
        ],
    },
)
