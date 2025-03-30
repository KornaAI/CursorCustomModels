# Tool Request Format Documentation

## Claude's Tool Request Format

When Claude (like me) needs to call a tool, it outputs a specific XML-like format. However, when this appears in Cursor, the client may interpret it as an actual tool call. Here's the format that Claude uses:

```
<function_calls>
<invoke name="tool_name">
<parameter name="param1">value1
</invoke>
</function_calls> 