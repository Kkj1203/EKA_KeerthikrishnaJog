"""
MCP.
This satisfies the MCP Server Integration requirement.
I use the official MCP Filesystem server and call its `write_file` tool from the 
Evaluator Agent to persist the final report to disk.
"""

import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _write_file_via_mcp(target_dir: str, filename: str, content: str) -> str:
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", target_dir],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            file_path = os.path.join(target_dir, filename)
            await session.call_tool(
                "write_file",
                arguments={"path": file_path, "content": content},
            )
            return file_path


def write_report_via_mcp(target_dir: str, filename: str, content: str) -> str:
    return asyncio.run(_write_file_via_mcp(target_dir, filename, content))