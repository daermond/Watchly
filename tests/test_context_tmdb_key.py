import asyncio

import pytest

from app.core.config import settings
from app.core.settings import get_default_settings
from app.models.library import LibraryCollection
from app.services.context import fetch_library_for_source
from app.services.profile import service as profile_module


@pytest.mark.parametrize("source", ["simkl", "trakt", "mdblist", "nuvio"])
@pytest.mark.parametrize("account_key,server_key", [("account-key", None), (None, "server-key")])
def test_external_library_uses_resolved_tmdb_key_and_account_language(monkeypatch, source, account_key, server_key):
    monkeypatch.setattr(settings, "TMDB_API_KEY", server_key)
    expected = LibraryCollection(source=source)
    calls = []
    real_factory = profile_module.get_tmdb_service

    def capture_factory(language="en-US", api_key=None):
        calls.append((language, api_key))
        return real_factory(language=language, api_key=api_key)

    async def external(self, configured_source, user_settings, token):
        assert configured_source == source and token == "test-account"
        return expected

    monkeypatch.setattr(profile_module, "get_tmdb_service", capture_factory)
    monkeypatch.setattr(profile_module.ProfileService, "fetch_external_library", external)
    user_settings = get_default_settings().model_copy(update={"tmdb_api_key": account_key, "language": "hu-HU"})
    result = asyncio.run(fetch_library_for_source(source, user_settings, "test-account", object(), None))
    assert result is expected
    assert calls == [("hu-HU", account_key or server_key)]
