import asyncio
import copy
import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import FastAPI

from app.api.endpoints import oauth
from app.api.models.tokens import TokenRequest
from app.core.security import STORED_SECRET_SENTINEL, mask_stored_secrets
from app.core.settings import UserSettings
from app.services.auth import auth_service
from app.services.profile.service import ProfileService
from app.services.simkl import SimklService


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(oauth.settings, "SIMKL_CLIENT_ID", "test-client")
    monkeypatch.setattr(oauth.settings, "SIMKL_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(oauth.settings, "HOST_NAME", "https://watchly.example")
    monkeypatch.setattr(oauth.settings, "APP_ENV", "production")
    app = FastAPI()
    app.include_router(oauth.router)
    return app


def test_pkce_matches_rfc7636_vector():
    assert oauth._pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == (
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


def test_https_authorize_and_callback_preserve_pkce_and_token_metadata(configured, monkeypatch):
    exchanges = []

    async def exchange(*args):
        exchanges.append(args)
        return {"access_token": "simkl_at_test", "refresh_token": "simkl_rt_test", "expires_in": 604800}

    async def user(*args):
        return {"user": {"name": "Test"}, "account": {"id": 123}}

    monkeypatch.setattr(oauth.simkl_service, "exchange_code", exchange)
    monkeypatch.setattr(oauth.simkl_service, "get_user_settings", user)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=configured), base_url="https://watchly.example"
        ) as c:
            redirect = await c.get("/auth/simkl")
            target = urlparse(redirect.headers["location"])
            assert target.scheme == "https" and target.netloc == "simkl.com" and target.path == "/oauth2/authorize"
            params = parse_qs(target.query)
            assert params["scope"] == ["media:read"] and params["code_challenge_method"] == ["S256"]
            verifier = c.cookies.get(oauth._SIMKL_PKCE_COOKIE)
            assert len(verifier) >= 43 and params["code_challenge"] == [oauth._pkce_challenge(verifier)]
            assert all("Secure" in s and "HttpOnly" in s for s in redirect.headers.get_list("set-cookie"))
            response = await c.get("/auth/simkl/callback", params={"code": "code", "state": params["state"][0]})
            assert response.status_code == 200
            assert "simkl_rt_test" in response.text and '"expires_at":' in response.text
            assert response.headers["cache-control"] == "no-store"
            assert not c.cookies.get(oauth._SIMKL_PKCE_COOKIE)
            assert exchanges == [
                ("code", "https://watchly.example/auth/simkl/callback", "test-client", "test-secret", verifier)
            ]

    asyncio.run(run())


@pytest.mark.parametrize("bad_state,missing_verifier", [(True, False), (False, True)])
def test_callback_rejects_bad_state_or_missing_pkce_before_exchange(
    configured, monkeypatch, bad_state, missing_verifier
):
    async def unexpected(*args):
        pytest.fail("Invalid callback reached SIMKL")

    monkeypatch.setattr(oauth.simkl_service, "exchange_code", unexpected)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=configured), base_url="https://watchly.example"
        ) as c:
            start = await c.get("/auth/simkl")
            state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
            if missing_verifier:
                c.cookies.delete(oauth._SIMKL_PKCE_COOKIE)
            result = await c.get(
                "/auth/simkl/callback", params={"code": "code", "state": "wrong" if bad_state else state}
            )
            assert result.status_code == 400

    asyncio.run(run())


def test_exchange_and_refresh_use_v2_and_code_is_never_retried():
    calls = []

    def transport(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"access_token": "new"})

    async def run():
        svc = SimklService()
        svc.client._client = httpx.AsyncClient(base_url=svc.base_url, transport=httpx.MockTransport(transport))
        await svc.exchange_code("code", "https://watchly.example/callback", "client", "secret", "verifier")
        await svc.refresh_token("refresh", "client", "secret")
        await svc.close()
        assert calls[0][0] == calls[1][0] == "/oauth2/token"
        assert calls[0][1]["code_verifier"] == "verifier"
        assert calls[1][1]["grant_type"] == "refresh_token"
        attempts = []

        def unavailable(request):
            attempts.append(request)
            return httpx.Response(503)

        svc.client._client = httpx.AsyncClient(base_url=svc.base_url, transport=httpx.MockTransport(unavailable))
        with pytest.raises(httpx.HTTPStatusError):
            await svc.exchange_code("code", "https://watchly.example/callback", "client", "secret", "verifier")
        assert len(attempts) == 1
        await svc.close()

    asyncio.run(run())


def test_refresh_metadata_roundtrips_and_refresh_secret_is_masked():
    saved = {"simkl_access_token": "access", "simkl_refresh_token": "refresh", "simkl_token_expires_at": 123456}
    masked = mask_stored_secrets(saved)
    assert masked["simkl_refresh_token"] == STORED_SECRET_SENTINEL
    restored = auth_service._build_user_settings(TokenRequest(**masked), saved)
    assert restored.simkl_refresh_token == "refresh"
    assert restored.simkl_token_expires_at == 123456


@pytest.fixture
def refresh_store(monkeypatch):
    from app.services.profile import service

    saved = {
        "settings": {
            "catalogs": [],
            "simkl_access_token": "old",
            "simkl_refresh_token": "refresh",
            "simkl_token_expires_at": 1,
        }
    }
    lock = asyncio.Lock()

    class Redis:
        def lock(self, *args, **kwargs):
            return lock

    async def get_client():
        return Redis()

    async def read(token, **kwargs):
        return copy.deepcopy(saved)

    async def write(token, data):
        saved.clear()
        saved.update(copy.deepcopy(data))

    monkeypatch.setattr(service.redis_service, "get_client", get_client)
    monkeypatch.setattr(service.token_store, "get_user_data", read)
    monkeypatch.setattr(service.token_store, "update_user_data", write)
    return saved


def test_concurrent_expired_requests_refresh_once_and_persist_pair(refresh_store, monkeypatch):
    from app.services.simkl import simkl_service

    calls = []

    async def refresh(*args):
        calls.append(args)
        await asyncio.sleep(0.01)
        return {"access_token": "new", "refresh_token": "refresh", "expires_in": 604800}

    monkeypatch.setattr(simkl_service, "refresh_token", refresh)

    async def run():
        svc = ProfileService.__new__(ProfileService)
        settings = [UserSettings(**refresh_store["settings"]) for _ in range(3)]
        result = await asyncio.gather(*(svc._refresh_simkl_token("account", s) for s in settings))
        assert result == ["new"] * 3 and len(calls) == 1

    asyncio.run(run())
    assert refresh_store["settings"]["simkl_access_token"] == "new"
    assert refresh_store["settings"]["simkl_refresh_token"] == "refresh"
    assert refresh_store["settings"]["simkl_token_expires_at"] > time.time()


def test_refresh_outage_preserves_saved_credentials(refresh_store, monkeypatch):
    from app.services.simkl import simkl_service

    original = copy.deepcopy(refresh_store)

    async def refresh(*args):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(simkl_service, "refresh_token", refresh)
    svc = ProfileService.__new__(ProfileService)
    result = asyncio.run(svc._fetch_simkl_history("account", UserSettings(**refresh_store["settings"])))
    assert result == (None, False) and refresh_store == original


def test_401_refreshes_and_retries_history_once(refresh_store, monkeypatch):
    from app.services.simkl import simkl_service

    refresh_store["settings"]["simkl_token_expires_at"] = int(time.time()) + 604800
    calls = []

    async def history(access, client):
        calls.append(access)
        if access == "old":
            req = httpx.Request("GET", "https://api.simkl.com/sync/all-items")
            raise httpx.HTTPStatusError("expired", request=req, response=httpx.Response(401, request=req))
        return "history"

    async def refresh(*args):
        return {"access_token": "new", "refresh_token": "refresh", "expires_in": 604800}

    monkeypatch.setattr(simkl_service, "refresh_token", refresh)
    monkeypatch.setattr(simkl_service, "get_history", history)
    svc = ProfileService.__new__(ProfileService)
    result = asyncio.run(svc._fetch_simkl_history("account", UserSettings(**refresh_store["settings"])))
    assert result == ("history", False) and calls == ["old", "new"]
