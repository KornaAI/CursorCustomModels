#!/usr/bin/env python3
"""
Generate System Prompt Script

This script generates a system prompt with proper XML tags for tool calls
that can be used with gemini_proxy_v2.py.
"""

def main():
    # Base system prompt - modify this as needed
    base_prompt = """
You are a powerful AI assistant with coding expertise.
Follow the user's instructions carefully and precisely.
"""

    # Tool format instructions with actual XML tags
    # Using string concatenation and raw strings to avoid issues with XML tags
    tool_format = r"""
# Tool Usage Instructions

You have access to several tools you can use to assist the user.
When you need to use a tool, you MUST format your request exactly as follows:
"""

    # Generate the examples with safer syntax that won't be cut off
    examples = """
Example for running a terminal command:
```xml
<function_calls>
<invoke name="run_terminal_cmd">
<parameter name="command">ls -la</parameter>
<parameter name="is_background">false</parameter>
</invoke>
</function_calls>
```

Example for reading a file:
```xml
<function_calls>
<invoke name="read_file">
<parameter name="target_file">path/to/file.txt</parameter>
<parameter name="offset">1</parameter>
<parameter name="limit">10</parameter>
</invoke>
</function_calls>
```

Example for editing a file:
```xml
<function_calls>
<invoke name="edit_file">
<parameter name="target_file">path/to/file.py</parameter>
<parameter name="instructions">Update the function to add error handling</parameter>
<parameter name="code_edit">def example():
    try:
        result = some_operation()
        return result
    except Exception as e:
        print(f"Error: {str(e)}")
        return None</parameter>
</invoke>
</function_calls>
```
"""

    # Final notes on tool usage
    notes = """
# Important Notes on Tool Usage:
1. Format the tool calls EXACTLY as shown
2. Do not modify the XML structure
3. Make sure all tags are properly closed
4. Don't include explanations within the tool call format
5. Don't use XML-like tags in your regular responses
"""

    # Combine all sections
    full_prompt = base_prompt + "\n" + tool_format + "\n" + examples + "\n" + notes

    # Save to a file
    with open("gemini_system_prompt_generated.txt", "w", encoding="utf-8") as f:
        f.write(full_prompt)
    
    print("System prompt generated and saved to 'gemini_system_prompt_generated.txt'")
    print("\nPreview of the system prompt:")
    print("=" * 80)
    print(full_prompt[:500] + "...\n(truncated)")
    print("=" * 80)

if __name__ == "__main__":
    main() 