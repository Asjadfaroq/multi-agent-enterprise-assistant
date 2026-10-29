from collections.abc import Callable

import psycopg
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool

from assistant.config import get_settings
from assistant.llm import get_embedding_model
from assistant.mcp_server.auth import (
    WRITE_TOOL_NAMES,
    Role,
    caller_user_id,
    can_write_tickets,
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
        server.tool()(tool)
    return server


def main() -> None:
    # Load the embedding model before serving so no caller pays for it on their first search.
    get_embedding_model()
    build_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()
