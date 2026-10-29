from collections.abc import Callable

import psycopg
from mcp.server.fastmcp import FastMCP

from assistant.config import get_settings
from assistant.mcp_server.auth import (
    WRITE_TOOL_NAMES,
    Role,
    can_write_tickets,
    current_user_id,
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


def build_server(role: Role) -> FastMCP:
    # Write tools are not even listed for roles that cannot use them, so the model is never
    # offered them; the tools themselves still re-check the role on every call.
    server = FastMCP("enterprise-assistant", log_level="WARNING")
    for tool in ALL_TOOLS:
        if tool.__name__ not in WRITE_TOOL_NAMES or can_write_tickets(role):
            server.tool()(tool)
    return server


def main() -> None:
    with psycopg.connect(get_settings().database_url) as connection:
        role = role_for_user(connection, current_user_id())
    build_server(role).run()


if __name__ == "__main__":
    main()
