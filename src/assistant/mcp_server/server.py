from collections.abc import Callable
from functools import wraps
from typing import Any

import psycopg
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from assistant.config import get_settings
from assistant.guardrails.pii import mask_pii_in_tool_result
from assistant.llm import get_embedding_model, get_reranker
from assistant.mcp_server.auth import (
    WRITE_TOOL_NAMES,
    Role,
    caller_user_id,
    can_write_tickets,
    configured_service_token,
    is_valid_service_authorization,
    role_for_user,
)
from assistant.mcp_server.tools_kb import search_documents
from assistant.mcp_server.tools_tickets import (
    create_ticket,
    get_ticket,
    list_tickets,
    update_ticket,
)

ALL_TOOLS: tuple[Callable[..., object], ...] = (
    search_documents,
    list_tickets,
    get_ticket,
    create_ticket,
    update_ticket,
)


def with_pii_masked_output(tool: Callable[..., Any]) -> Callable[..., Any]:
    # Masking at the source means neither the model, the checkpoint nor a trace sees raw PII from
    # documents or tickets. `wraps` keeps the signature FastMCP builds the tool schema from.
    @wraps(tool)
    def masked_tool(*args: Any, **kwargs: Any) -> Any:
        return mask_pii_in_tool_result(tool(*args, **kwargs))

    return masked_tool


def tools_visible_to_role(tools: list[MCPTool], role: Role) -> list[MCPTool]:
    return [tool for tool in tools if tool.name not in WRITE_TOOL_NAMES or can_write_tickets(role)]


class RoleAwareServer(FastMCP):
    # Write tools are not even listed to roles that cannot use them, so the model is never offered
    # them; the tools themselves still re-check the caller's role on every call.
    async def list_tools(self) -> list[MCPTool]:
        caller_id = caller_user_id(self.get_context())
        with psycopg.connect(get_settings().database_url) as connection:
            role = role_for_user(connection, caller_id)
        return tools_visible_to_role(await super().list_tools(), role)


def build_server() -> RoleAwareServer:
    settings = get_settings()
    server = RoleAwareServer(
        "enterprise-assistant",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
        stateless_http=True,
        json_response=True,
        log_level="WARNING",
    )
    for tool in ALL_TOOLS:
        server.tool()(with_pii_masked_output(tool))
    return server


def require_service_token(app: ASGIApp, expected_token: str) -> ASGIApp:
    async def reject_requests_without_service_token(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        # Only HTTP requests carry headers; lifespan events must pass so the MCP app can start.
        if scope["type"] == "http" and not is_valid_service_authorization(
            Headers(scope=scope).get("authorization"), expected_token
        ):
            response = JSONResponse({"detail": "Missing or invalid service token"}, status_code=401)
            await response(scope, receive, send)
            return
        await app(scope, receive, send)

    return reject_requests_without_service_token


def build_http_app() -> ASGIApp:
    return require_service_token(build_server().streamable_http_app(), configured_service_token())


def main() -> None:
    settings = get_settings()
    http_app = build_http_app()
    # Load both models before serving so no caller pays for loading them on their first search.
    get_embedding_model()
    get_reranker()
    uvicorn.run(
        http_app, host=settings.mcp_server_host, port=settings.mcp_server_port, log_level="warning"
    )


if __name__ == "__main__":
    main()
