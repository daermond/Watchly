import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.models.tokens import TokenRequest
from app.core.app import app
from app.core.config import settings as server_settings
from app.core.settings import UserSettings, get_default_settings, settings_from_credentials
from app.models.library import LibraryCollection
from app.services.language_service import fetch_languages_list, get_countries_list
from app.services.recommendation.catalog_service import catalog_service
from app.services.recommendation.filtering import apply_discover_filters, matches_details_preferences
from app.services.recommendation.metadata import RecommendationMetadata
from app.services.redis_service import RedisService, redis_service
from app.services.token_store import TokenStore, token_store
from app.services.user_cache import USER_CACHE_TTL_SECONDS, UserCacheService


def test_options_work_without_server_key(monkeypatch):
    monkeypatch.setattr(server_settings, "TMDB_API_KEY", None)
    languages = asyncio.run(fetch_languages_list())
    assert len(languages) > 100
    assert any(item["iso_639_1"] == "hu-HU" for item in languages)
    assert any(item["iso_3166_1"] == "HU" for item in get_countries_list())
    # Callers cannot change the shared reference data.
    languages[0]["language"] = "changed"
    assert asyncio.run(fetch_languages_list())[0]["language"] != "changed"


def test_configure_shows_language_and_optional_country_choices(monkeypatch):
    monkeypatch.setattr(token_store, "count_users", AsyncMock(return_value=1))
    response = TestClient(app).get("/configure")
    assert response.status_code == 200
    assert 'value="hu-HU"' in response.text
    assert 'id="countrySelect"' in response.text
    assert 'value="HU"' in response.text


def test_countries_validate_normalize_and_roundtrip():
    request = TokenRequest(selected_countries=["hu", " HU ", "US"], language="hu-HU")
    assert request.selected_countries == ["HU", "US"]
    stored = UserSettings(catalogs=[], **request.model_dump(exclude={"catalogs"}))
    assert settings_from_credentials({"settings": stored.model_dump()}).selected_countries == ["HU", "US"]
    assert get_default_settings().selected_countries == []
    with pytest.raises(ValidationError):
        TokenRequest(selected_countries=["XYZ"])


def test_country_discover_override_and_authoritative_detail_filter():
    settings = get_default_settings()
    settings.selected_countries = ["HU"]
    settings.excluded_movie_genres = ["27"]
    assert apply_discover_filters({"with_origin_country": "US"}, settings)["with_origin_country"] == "HU"
    assert matches_details_preferences({"production_countries": [{"iso_3166_1": "HU"}]}, settings, "movie")
    assert matches_details_preferences({"origin_country": ["HU", "US"]}, settings, "series")
    assert not matches_details_preferences({"origin_country": ["US"]}, settings, "movie")
    assert not matches_details_preferences({"origin_country": ["HU"], "genres": [{"id": 27}]}, settings, "movie")
    assert not matches_details_preferences({}, settings, "movie")
    assert matches_details_preferences({}, get_default_settings(), "movie")


def test_metadata_filters_sparse_candidates_before_images():
    settings = get_default_settings()
    settings.selected_countries = ["HU"]
    settings.excluded_movie_genres = ["27"]
    details = {
        1: {"id": 1, "title": "Keep", "origin_country": ["HU"], "genres": [{"id": 18}]},
        2: {"id": 2, "title": "Wrong country", "origin_country": ["US"], "genres": []},
        3: {"id": 3, "title": "Excluded", "origin_country": ["HU"], "genres": [{"id": 27}]},
    }
    tmdb = SimpleNamespace(
        get_movie_details=AsyncMock(side_effect=lambda tid: details[tid]),
        get_images_for_title=AsyncMock(return_value={}),
    )
    result = asyncio.run(RecommendationMetadata.fetch_batch(tmdb, [{"id": i} for i in details], "movie", settings))
    assert [item["id"] for item in result] == ["tmdb:1"]
    tmdb.get_images_for_title.assert_awaited_once_with("movie", 1, language="en-US")


@pytest.mark.parametrize("catalog_id", ["watchly.rec", "watchly.creators"])
def test_trending_fallback_excludes_watched_imdb_and_tmdb(monkeypatch, catalog_id):
    monkeypatch.setattr(
        catalog_service,
        "_get_trending_fallback",
        AsyncMock(
            return_value=[
                {"id": "tt1", "_tmdb_id": 1},
                {"id": "tt2", "_tmdb_id": 2},
                {"id": "tt3", "_tmdb_id": 3},
            ]
        ),
    )
    result = asyncio.run(
        catalog_service._get_recommendations(
            catalog_id,
            "movie",
            {"creators": object()},
            None,
            {2},
            {"tt1"},
            LibraryCollection(),
            20,
            get_default_settings(),
        )
    )
    assert [item["id"] for item in result] == ["tt3"]


def test_rewatch_row_keeps_watched_titles():
    service = SimpleNamespace(get_rewatch_picks=AsyncMock(return_value=[{"id": "tt1", "_tmdb_id": 1}]))
    result = asyncio.run(
        catalog_service._get_recommendations(
            "watchly.rewatch",
            "movie",
            {"rewatch": service},
            None,
            {1},
            {"tt1"},
            LibraryCollection(),
            20,
            get_default_settings(),
        )
    )
    assert result == [{"id": "tt1", "_tmdb_id": 1}]


def test_cipher_reused_and_changes_with_salt(monkeypatch):
    store = TokenStore()
    monkeypatch.setattr(server_settings, "TOKEN_SALT", "first-test-salt")
    first = store._get_cipher()
    assert store._get_cipher() is first
    encrypted = store.encrypt_token("value")
    assert store.decrypt_token(encrypted) == "value"
    monkeypatch.setattr(server_settings, "TOKEN_SALT", "second-test-salt")
    assert store._get_cipher() is not first
    monkeypatch.setattr(server_settings, "TOKEN_SALT", "change-me")
    with pytest.raises(RuntimeError):
        store._get_cipher()


def test_getex_refreshes_expiry_in_one_call(monkeypatch):
    client = SimpleNamespace(getex=AsyncMock(return_value="cached"))
    service = RedisService()
    monkeypatch.setattr(service, "get_client", AsyncMock(return_value=client))
    assert asyncio.run(service.getex("test", 100)) == "cached"
    client.getex.assert_awaited_once_with("test", ex=100)


def test_watched_cache_uses_getex_without_separate_expire(monkeypatch):
    from app.services import cache_codec

    reader = AsyncMock(return_value=cache_codec.encode('{"watched_tmdb":[1],"watched_imdb":["tt1"]}'))
    expire = AsyncMock()
    monkeypatch.setattr(redis_service, "getex", reader)
    monkeypatch.setattr(redis_service, "expire", expire)
    assert asyncio.run(UserCacheService().get_watched_sets("test", "movie")) == ({1}, {"tt1"})
    assert reader.await_args.args[1] == USER_CACHE_TTL_SECONDS
    expire.assert_not_awaited()


def test_count_users_cached_within_process(monkeypatch):
    calls = []

    async def scan(**kwargs):
        calls.append(kwargs)
        yield "user1"
        yield "user2"

    store = TokenStore()
    monkeypatch.setattr(redis_service, "get_client", AsyncMock(return_value=SimpleNamespace(scan_iter=scan)))

    async def run():
        assert await store.count_users() == 2
        assert await store.count_users() == 2
        store.count_users.cache_clear()

    asyncio.run(run())
    assert len(calls) == 1


def test_theme_row_keeps_more_than_twenty_with_existing_work_cap(monkeypatch):
    from app.services.recommendation.theme_based import ThemeBasedService

    service = ThemeBasedService(SimpleNamespace(), get_default_settings())
    candidates = [{"id": i, "genre_ids": [18], "vote_average": 7} for i in range(1, 81)]
    monkeypatch.setattr(service, "_fetch_discover_candidates", AsyncMock(return_value=candidates))
    enriched = AsyncMock(side_effect=lambda tmdb, items, content_type, user_settings: items)
    monkeypatch.setattr(RecommendationMetadata, "fetch_batch", enriched)
    result = asyncio.run(service.get_recommendations_for_theme("watchly.theme.a:g18", "movie"))
    assert len(result) == 40
    assert len({item["id"] for item in result}) == 40


def test_country_language_save_keeps_preferences_and_invalidates_old_rows(monkeypatch):
    from test_account_identity import setup_fakes

    from app.services.auth import AuthService
    from app.services.user_cache import user_cache

    setup_fakes(monkeypatch)
    invalidation = AsyncMock()
    monkeypatch.setattr(user_cache, "invalidate_all_catalogs", invalidation)
    service = AuthService()

    async def run():
        first, _, first_settings = await service.create_user_token(
            TokenRequest(
                trakt_access_token="test-token",
                watch_history_source="trakt",
                language="hu-HU",
                selected_countries=["HU"],
            )
        )
        assert first_settings.selected_countries == ["HU"]
        second, _, second_settings = await service.create_user_token(
            TokenRequest(
                trakt_access_token="test-token",
                watch_history_source="trakt",
                language="hu-HU",
                selected_countries=["US"],
            )
        )
        assert first.token == second.token
        assert second_settings.selected_countries == ["US"]
        saved = await token_store.get_user_data(first.token, fresh=True)
        assert saved["settings"]["language"] == "hu-HU"
        assert saved["settings"]["selected_countries"] == ["US"]
        invalidation.assert_awaited_once_with(first.token)

    asyncio.run(run())
