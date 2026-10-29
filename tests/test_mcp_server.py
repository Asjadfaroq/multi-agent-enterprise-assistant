import asyncio

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from starlette.responses import PlainTextResponse
from starlette.types import Receive, Scope, Send

from assistant.mcp_server.auth import configured_service_token, service_authorization_header
from assistant.mcp_server.server import build_http_app, build_server, require_service_token

SERVICE_TOKEN = "test-service-token"


async def reachable_app(scope: Scope, receive: Receive, send: Send) -> None:
    await PlainTextResponse("reached the MCP app")(scope, receive, send)


def post_to(app: object, headers: dict[str, str]) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mcp") as client:
            return await client.post("/mcp", headers=headers, json={})

    return asyncio.run(send())


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": service_authorization_header("wrong-token")},
        {"Authorization": SERVICE_TOKEN},
    ],
    ids=["missing", "wrong", "missing-bearer-scheme"],
)
def test_request_without_the_service_token_is_rejected_with_401(headers: dict[str, str]) -> None:
    response = post_to(require_service_token(reachable_app, SERVICE_TOKEN), headers)

    assert response.status_code == 401


def test_request_with_the_service_token_reaches_the_mcp_app() -> None:
    headers = {"Authorization": service_authorization_header(SERVICE_TOKEN), "X-User": "nora"}

    response = post_to(require_service_token(reachable_app, SERVICE_TOKEN), headers)

    assert response.status_code == 200 and response.text == "reached the MCP app"


def test_lifespan_events_pass_through_so_the_mcp_app_can_start() -> None:
    received_scope_types: list[str] = []

    async def recording_app(scope: Scope, receive: Receive, send: Send) -> None:
        received_scope_types.append(scope["type"])

    asyncio.run(
        require_service_token(recording_app, SERVICE_TOKEN)({"type": "lifespan"}, None, None)
    )

    assert received_scope_types == ["lifespan"]


def test_real_server_app_rejects_a_caller_that_only_sends_x_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "assistant.mcp_server.server.configured_service_token", lambda: SERVICE_TOKEN
    )

    response = post_to(build_http_app(), {"X-User": "lina"})

    assert response.status_code == 401


def test_missing_service_token_setting_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "")
    from assistant.config import get_settings

    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="MCP_SERVICE_TOKEN is not set"):
            configured_service_token()
    finally:
        get_settings.cache_clear()


def test_pii_masking_wrapper_keeps_the_tool_schema_the_model_sees() -> None:
    tools = {tool.name: tool for tool in asyncio.run(FastMCP.list_tools(build_server()))}

    search_parameters = tools["search_documents"].inputSchema["properties"]

    assert set(search_parameters) == {"query", "k"}
