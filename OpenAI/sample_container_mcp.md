# Container MCP Server Example

This is an incomplete example showing how to implement a container tool for GPT using MCP.

```
from mcp.server.fastmcp import fastmcp
# dummy showing how to import container tool
from swe_rex import SweRexManager

# Pass lifespan to server
mcp = FastMCP(
    name="container",
    instructions=r"""
Utilities for interacting with a container, for example, a Docker container.\n
(container_tool, version 1.X.X)\n
(lean_terminal, version 1.X.X)
""".strip(),
)

swe_rex_manager = SweRexManager()

def _get_session_id(ctx: Context) -> str:
    """Extract session ID from headers, URL query parameter or fallback to client_id"""
    request = ctx.request_context.request
    return request.headers.get("session_id") or request.query_params.get(
        "session_id"
    ) or ctx.client_id

@mcp.tool(
    name="exec",
    title="container exec",
    description="""
Returns the output of the command.
Allocates an interactive pseudo-TTY if (and only if) 'session_name' is set.
    """,
    )
async def exec(
    ctx: Context,
    cmd: list[str],
    session_name: Optional[str] = None,
    workdir: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[dict[str, str]] = None,
    user: Optional[str] = None,
) -> str:
    session_id = _get_session_id(ctx)
    try:
        logger.debug(f"cmd for container exec: {cmd}")

        res = await swe_rex_manager.execute_in_session(
            session_id,
            cmd=cmd,
            workdir=workdir,
            env=env,
            execution_timeout=360 if timeout is None else timeout,
            # Below fields are not used right now
            session_name=session_name, # This could be overriding session_id
            user=user,
        )
        logger.info(f"container execution result: {res}")
        return res

@mcp.tool(
    name="cleanup_session",
    title="clean container session",
    description="cleanup a specific session",
    annotations={
        "include_in_prompt": False,
    })
async def cleanup_session(ctx: Context) -> None:
    """Cleanup a specific session"""
    session_id = _get_session_id(ctx)
    logger.info(f"Cleaning up session: {session_id}")
    await swe_rex_manager.cleanup_session(session_id)
```

### SweRexManager Implementation Pattern

Based on the RemoteRuntime pattern, your SweRexManager could be implemented like below
Note that this is a dummy implementation and you should implement your own version.
```
from typing import Dict, Any, Optional
import asyncio
from swerex.runtime.remote import RemoteRuntime
from swerex.runtime.config import RemoteRuntimeConfig

class SweRexManager:
    def __init__(self, config: Dict[str, Any]):
        """Initialize SweRexManager with dict configuration.

        Args:
            config: Dictionary containing:
                - host: Server host (required)
                - port: Server port (optional)
                - timeout: Request timeout in seconds (optional, default 30.0)
                - auth_token: Authentication token (optional)
        """
        self.config = RemoteRuntimeConfig(**config)
        self.runtime = RemoteRuntime.from_config(self.config)
        self.sessions: Dict[str, str] = {}  # session_id -> runtime_session mapping

    async def execute_in_session(
        self,
        session_id: str,
        cmd: list[str],
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        execution_timeout: int = 360,
        **kwargs
    ) -> str:
        """Execute command in a session."""
        # Ensure session exists
        if session_id not in self.sessions:
            await self.create_session(session_id)

        from swerex.runtime.abstract import Command

        command = Command(
            command=cmd,
            timeout=execution_timeout,
            cwd=workdir,
            env=env or {}
        )

        response = await self.runtime.execute(command)
        return response.stdout if response.exit_code == 0 else response.stderr

    async def create_session(self, session_id: str) -> None:
        """Create a new session."""
        from swerex.runtime.abstract import CreateSessionRequest

        request = CreateSessionRequest(session_id=session_id)
        await self.runtime.create_session(request)
        self.sessions[session_id] = session_id

    async def cleanup_session(self, session_id: str) -> None:
        """Cleanup a session."""
        if session_id in self.sessions:
            from swerex.runtime.abstract import CloseSessionRequest

            request = CloseSessionRequest(session_id=session_id)
            await self.runtime.close_session(request)
            del self.sessions[session_id]
```
