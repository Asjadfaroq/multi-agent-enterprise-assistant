from mcp.server.fastmcp import FastMCP

from assistant.mcp_server.tools_kb import search_documents

server = FastMCP("enterprise-assistant")
server.tool()(search_documents)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
