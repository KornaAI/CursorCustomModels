# Tool Request Format Documentation for Gemini Proxy

## How @gemini_proxy_v2.py Handles Tool Requests

The `extract_tool_call_requests` function in the Gemini proxy v2 script uses regex pattern matching to detect when a model is trying to use a tool/function. This is important for the auto-continuation feature.

### Current Regex Patterns for Tool Detection

The proxy looks for these patterns in the text:

1. Claude-style XML format (shown with escaping):
   ```
   <function_calls>...<invoke name="TOOL_NAME">...</invoke>...</function_calls>
   ```

2. JSON format in code blocks:
   ```
   ```json
   {
     "action": "TOOL_NAME",
     ...
   }
   ```

3. Simple text mention:
   ```
   I need to use the TOOL_NAME tool
   ```

### Implementation in the Code

From `gemini_proxy_v2.py`, the function that handles this detection:

```python
def extract_tool_call_requests(content):
    """Extract tool call requests from agent's text output"""
    # Simple pattern matching to detect tool call requests in the agent's output
    tool_call_patterns = [
        r'<function_calls>[\s\S]*?<invoke name="([^"]+)">([\s\S]*?)<\/antml:invoke>[\s\S]*?<\/antml:function_calls>',
        r'```(json)?[\s\n]*\{\s*"action"\s*:\s*"([^"]+)"[\s\S]*?\}[\s\n]*```',
        r'I need to use the ([a-zA-Z_]+) tool'
    ]
    
    for pattern in tool_call_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        if matches:
            if isinstance(matches[0], tuple):
                # For the detailed pattern that captures more info
                tool_name = matches[0][0]
            else:
                # For the simpler pattern
                tool_name = matches[0]
                
            logger.info(f"Detected potential tool call request for tool: {tool_name}")
            return True, tool_name
            
    return False, None
```

### Auto-Continuation Process

When a tool call is detected:
1. The tool name is stored in the `last_tool_call` cache
2. The `continuation_counter` is used to track conversation turns
3. If a function result comes back, a system message is appended:
   ```
   "Continue from where you left off after using the tool. Process the tool results and proceed with the next steps to complete the task."
   ```

This approach allows the model to continue its work after tool usage without requiring additional user input. 