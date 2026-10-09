import json
from pathlib import Path

_OPTIONS = json.loads(Path(__file__).with_name("tmdb-options.json").read_text(encoding="utf-8-sig"))


async def fetch_languages_list() -> list[dict[str, str]]:
    return [dict(item) for item in _OPTIONS["languages"]]


def get_countries_list() -> list[dict[str, str]]:
    return sorted((dict(item) for item in _OPTIONS["countries"]), key=lambda item: item["english_name"])
