"""MCP tool encapsulation for the agent runtime (Module 6).

Wraps the runtime's capabilities as MCP-style tool descriptors so they can be
exposed to an MCP client or gateway. Every tool is classified:

* ``read``  tools run **synchronously** inside the workflow (no side effects).
* ``write`` tools are gated: their ``execution_mode`` is
  ``async_after_approval`` — the workflow only *proposes* them, and once a
  human approves, they are executed by the :class:`AsyncToolQueue` so they
  never block the user's streaming reply.

Two concrete write tools requested by the spec are provided as canonical
examples: Markdown export and meeting transcription.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

READ = "read"
WRITE = "write"
EXEC_SYNC = "sync"
EXEC_ASYNC = "async_after_approval"


@dataclass
class MCPTool:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    kind: str = READ
    execution_mode: str = EXEC_SYNC

    def to_mcp(self) -> dict[str, Any]:
        """Serialize to the MCP ``tools/list`` entry shape."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {"kind": self.kind, "execution_mode": self.execution_mode},
        }


@dataclass
class MCPToolRegistry:
    _tools: dict[str, MCPTool] = field(default_factory=dict)

    def register(self, tool: MCPTool) -> None:
        # Validate the tool's input schema at registration time so a malformed
        # or unsafe tool (e.g. a synchronous write) fails fast instead of
        # surfacing a broken descriptor to MCP clients.
        self._validate_schema(tool)
        self._tools[tool.name] = tool

    @staticmethod
    def _validate_schema(tool: MCPTool) -> None:
        """Validate an MCP tool's ``input_schema`` contract (Module 6).

        Raises ``ValueError`` when the schema is not a JSON-Schema ``object``,
        declares ``required`` fields that are absent from ``properties``, or a
        ``write`` tool is configured to execute synchronously (it must use
        ``async_after_approval`` so it is never applied without human approval).
        """
        schema = tool.input_schema
        if not isinstance(schema, dict):
            raise ValueError(f"tool '{tool.name}': input_schema must be a dict")
        if schema.get("type") != "object":
            raise ValueError(f"tool '{tool.name}': input_schema.type must be 'object'")
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"tool '{tool.name}': input_schema.properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise ValueError(f"tool '{tool.name}': input_schema.required must be a list")
        missing = set(required) - set(properties.keys())
        if missing:
            raise ValueError(
                f"tool '{tool.name}': required fields {sorted(missing)} missing from properties"
            )
        if tool.kind == WRITE and tool.execution_mode == EXEC_SYNC:
            raise ValueError(
                f"tool '{tool.name}': write tools must use async_after_approval execution mode"
            )

    def get(self, name: str) -> MCPTool | None:
        return self._tools.get(name)

    def tool_list(self) -> list[dict[str, Any]]:
        return [t.to_mcp() for t in self._tools.values()]

    def read_tools(self) -> list[MCPTool]:
        return [t for t in self._tools.values() if t.kind == READ]

    def write_tools(self) -> list[MCPTool]:
        return [t for t in self._tools.values() if t.kind == WRITE]


# --- Canonical tools (examples required by the spec) ---------------------- #

MARKDOWN_WRITE = MCPTool(
    name="write_markdown_document",
    title="Write Markdown Document",
    description="Persist a Markdown document (e.g. meeting notes) to the knowledge base.",
    input_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "conversation_id": {"type": "integer"},
        },
        "required": ["title", "content"],
    },
    kind=WRITE,
    execution_mode=EXEC_ASYNC,
)

MEETING_TRANSCRIBE = MCPTool(
    name="transcribe_meeting_recording",
    title="Transcribe Meeting Recording",
    description="Trigger asynchronous transcription of a meeting recording file.",
    input_schema={
        "type": "object",
        "properties": {
            "recording_file_id": {"type": "integer"},
            "language": {"type": "string"},
        },
        "required": ["recording_file_id"],
    },
    kind=WRITE,
    execution_mode=EXEC_ASYNC,
)

QUERY_CONTEXT = MCPTool(
    name="query_context_chunks",
    title="Query Context Chunks",
    description="Retrieve relevant context chunks for the conversation (read-only).",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    kind=READ,
    execution_mode=EXEC_SYNC,
)


def default_registry() -> MCPToolRegistry:
    """A starting registry mirroring the runtime's read/write tool split."""
    reg = MCPToolRegistry()
    reg.register(QUERY_CONTEXT)
    reg.register(MARKDOWN_WRITE)
    reg.register(MEETING_TRANSCRIBE)
    return reg
