For gpt expected container tool, here's an incomplete example
Note that the SweRexManager or swe_rex are dummies, you need to implement your own container tool with session management
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
