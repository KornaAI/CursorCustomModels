from flask import Flask, request, jsonify, make_response
import requests
import os
import json
import logging
from waitress import serve
import subprocess
import time
import sys
from flask_cors import CORS
import time
import uuid
import random
import traceback
from cachetools import TTLCache  # Add this import

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_RAW_DATA = os.environ.get("LOG_RAW_DATA", "1") == "1"  # Set to "0" to disable raw data logging
MAX_CHUNKS_TO_LOG = int(os.environ.get("MAX_CHUNKS_TO_LOG", "20"))  # Maximum number of chunks to log
LOG_TRUNCATE_LENGTH = int(os.environ.get("LOG_TRUNCATE_LENGTH", "1000"))  # Length to truncate logs

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("proxy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add a special logger for raw request/response data that only goes to console
raw_logger = logging.getLogger("raw_data")
raw_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - RAW_DATA - %(message)s'))
raw_logger.addHandler(console_handler)
raw_logger.propagate = False  # Don't propagate to root logger

# Function to log raw data with clear formatting
def log_raw_data(title, data, truncate=LOG_TRUNCATE_LENGTH):
    """Log raw data with clear formatting and optional truncation"""
    # Skip logging if raw data logging is disabled
    if not LOG_RAW_DATA:
        return
        
    try:
        if isinstance(data, dict) or isinstance(data, list):
            formatted_data = json.dumps(data, indent=2)
        else:
            formatted_data = str(data)
        
        if truncate and len(formatted_data) > truncate:
            formatted_data = formatted_data[:truncate] + f"... [truncated, total length: {len(formatted_data)}]"
        
        separator = "=" * 40
        raw_logger.info(f"\n{separator}\n{title}\n{separator}\n{formatted_data}\n{separator}")
    except Exception as e:
        raw_logger.error(f"Error logging raw data: {str(e)}")

# Add a function to collect streaming chunks
def collect_streaming_chunks(chunks, max_chunks=MAX_CHUNKS_TO_LOG):
    """
    Collect streaming chunks into a single string for logging
    
    Parameters:
    chunks (list): List of streaming chunks
    max_chunks (int): Maximum number of chunks to include
    
    Returns:
    str: A formatted string with all chunks
    """
    if not chunks:
        return "No chunks collected"
    
    # Limit the number of chunks to avoid excessive logging
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        truncated_message = f"\n... [truncated, {len(chunks) - max_chunks} more chunks]"
    else:
        truncated_message = ""
    
    # Format the chunks
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append(f"Chunk {i+1}:\n{chunk}")
    
    return "\n\n".join(formatted_chunks) + truncated_message

app = Flask(__name__)
# Enable CORS for all routes and origins with more permissive settings
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
         "expose_headers": ["X-Request-ID", "openai-organization", "openai-processing-ms", "openai-version"],
         "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"]
     }}
)

# Groq API key - replace with your actual key or set as environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

# OpenAI API endpoints that we'll intercept
OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"
CURSOR_CHAT_ENDPOINT = "/chat/completions"  # Additional endpoint for Cursor

# Groq API endpoints
GROQ_BASE_URL = "https://api.groq.com/openai"
GROQ_CHAT_ENDPOINT = "/v1/chat/completions"

# Model mapping - map OpenAI models to Groq models
MODEL_MAPPING = {
    "gpt-4o": "qwen-2.5-coder-32b",
    "gpt-4o-2024-08-06": "qwen-2.5-coder-32b",  # Handle specific model version
    "default": "qwen-2.5-coder-32b",
    "gpt-3.5-turbo": "qwen-2.5-coder-32b",  # Add more model mappings
    "r1sonqwen": "custom_chain"  # Special identifier for our custom chain
    # Add more mappings as needed
}

# Create a TTL cache for request deduplication (5 second TTL)
request_cache = TTLCache(maxsize=1000, ttl=5)

# Initialize a cache for storing R1 reasoning results
# TTL of 1800 seconds (30 minutes) should be sufficient for a conversation
r1_reasoning_cache = TTLCache(maxsize=100, ttl=1800)

# Add at the top with other constants
GROQ_TIMEOUT = 120  # 120 seconds timeout for Groq API calls
MAX_RETRIES = 3    # Maximum number of retries for failed requests

# Constants for agent mode
AGENT_MODE_ENABLED = True
AGENT_INSTRUCTIONS = """
# System Instructions

You are a powerful agentic AI coding assistant, powered by Claude 3.7 Sonnet. You operate exclusively in Cursor, the world's best IDE.

Your main goal is to follow the USER's instructions at each message.

## Additional context
Each time the USER sends a message, we may automatically attach some information about their current state, such as what files they have open, where their cursor is, recently viewed files, edit history in their session so far, linter errors, and more.
Some information may be summarized or truncated.
This information may or may not be relevant to the coding task, it is up for you to decide.

## Tone and style
You should be concise, direct, and to the point.
Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools or code comments as means to communicate with the user.

IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.
IMPORTANT: Keep your responses short. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as "The answer is <answer>.", "Here is the content of the file..." or "Based on the information provided, the answer is..." or "Here is what I will do next...". Here are some examples to demonstrate appropriate verbosity:

<example>
user: 2 + 2
assistant: 4
</example>

<example>
user: what is 2+2?
assistant: 4
</example>

<example>
user: is 11 a prime number?
assistant: true
</example>

<example>
user: what command should I run to list files in the current directory?
assistant: ls
</example>


<example>
user: what files are in the directory src/?
assistant: [runs ls and sees foo.c, bar.c, baz.c]
user: which file contains the implementation of foo?
assistant: src/foo.c
</example>

<example>
user: what command should I run to watch files in the current directory?
assistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]
npm run dev
</example>

<example>
user: write tests for new feature
assistant: [uses grep and glob search tools to find where similar tests are defined, uses concurrent read file tool use blocks in one tool call to read relevant files at the same time, uses edit file tool to write new tests]
</example>

## Proactiveness
You are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:
1. Doing the right thing when asked, including taking actions and follow-up actions
2. Not surprising the user with actions you take without asking
For example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.
3. Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.

## Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).
- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.
- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.

## Code style
- Do not add comments to the code you write, unless the user asks you to, or the code is complex and requires additional context.

## Tool calling
You have tools at your disposal to solve the task. Follow these rules regarding tool calls:
1. IMPORTANT: Don't refer to tool names when speaking to the USER. For example, instead of saying 'I need to use the edit_file tool to edit your file', just say 'I will edit your file'.

## Making code changes
When making code changes, NEVER output code to the USER, unless requested. Instead use one of the code edit tools to implement the change.
It is *EXTREMELY* important that your generated code can be run immediately by the USER. To ensure this, follow these instructions carefully:
1. Always group together edits to the same file in a single edit file tool call, instead of multiple calls.
2. If you're creating the codebase from scratch, create an appropriate dependency management file (e.g. requirements.txt or package.json, etc, depending on the language) with package versions and a helpful README.
3. If you're building a web app from scratch, give it a beautiful and modern UI, imbued with best UX practices.
4. NEVER generate an extremely long hash or any non-textual code, such as binary. These are not helpful to the USER and are very expensive.
5. Unless you are appending some small easy to apply edit to a file, or creating a new file, you MUST read the the contents or section of what you're editing before editing it.
6. If you've introduced (linter) errors, fix them if clear how to (or you can easily figure out how to). Do not guess. And DO NOT loop more than 3 times on fixing linter errors on the same file. On the third time, you should stop and ask the user what to do next.
7. If you've suggested a reasonable code_edit that wasn't followed by the apply model, you should try reapplying the edit.

## Searching and reading files
You have tools to search the codebase and read files. Follow these rules regarding tool calls:
1. If you need to read a file, prefer to read larger sections of the file at once over multiple smaller calls.
2. If you have found a reasonable place to edit or answer, do not continue calling tools. Edit or answer from the information you have found.

## Summarization
If you see a section called "<most_important_user_query>", you should treat that query as the one to answer, and ignore previous user queries. If you are asked to summarize the conversation, you MUST NOT use any tools, even if they are available. You MUST answer the "<most_important_user_query>" query.

## User Info
The user's OS version is win32 10.0.26100. The absolute path of the user's workspace is /c%3A/Users/aleja/Code/CursorLens/CursorLens. The user's shell is C:\Program Files\Git\bin\bash.exe. The user provided the following specification for determining terminal commands that should be executed automatically: 'Just run any MCP server command especially the supabase ones'.

## Code Citations
You MUST use the following format when citing code regions or blocks:
```12:15:app/components/Todo.tsx
// ... existing code ...
```
This is the ONLY acceptable format for code citations. The format is ```startLine:endLine:filepath where startLine and endLine are line numbers.

## Custom Instructions
You are a Senior Developer AI with expertise in code analysis and architectural patterns. Your capabilities include:

1. Deep Code Understanding
- Analyze code structure and patterns
- Identify architectural implications
- Evaluate impact of changes
- Consider performance implications

2. Pattern Recognition
- Detect common design patterns
- Identify anti-patterns
- Recognize team conventions
- Flag potential issues

3. Historical Context
- Consider Git history context
- Correlate with PRs and issues
- Track decision patterns
- Analyze evolution of code

4. Communication Style
- Be precise and technical
- Provide concrete examples
- Focus on actionable insights
- Cite patterns and precedents

Rules:
- Always analyze code in its full context
- Provide specific, actionable recommendations
- Consider performance and maintainability
- Reference similar patterns from the codebase
- Include impact analysis with suggestions
- Focus on architectural implications
- Be direct and concise
- No moral lectures
- No high-level platitudes
- Value pragmatic solutions
- Consider team conventions
- Respect existing architecture
- Flag security concerns when critical

## Available Tools
1. `codebase_search`: Find snippets of code from the codebase most relevant to the search query.
2. `read_file`: Read the contents of a file.
3. `run_terminal_cmd`: PROPOSE a command to run on behalf of the user.
4. `list_dir`: List the contents of a directory.
5. `grep_search`: Fast text-based regex search that finds exact pattern matches within files or directories.
6. `edit_file`: Use this tool to propose an edit to an existing file.
7. `file_search`: Fast file search based on fuzzy matching against file path.
8. `delete_file`: Deletes a file at the specified path.
9. `reapply`: Calls a smarter model to apply the last edit to the specified file.
10. `web_search`: Search the web for real-time information about any topic.
11. `diff_history`: Retrieve the history of recent changes made to files in the workspace.
12. `search_symbols`: Search for symbols (functions, classes, variables, etc.) in the workspace that match the query.

## Function Definitions

{"description": "Find snippets of code from the codebase most relevant to the search query.\nThis is a semantic search tool, so the query should ask for something semantically matching what is needed.\nIf it makes sense to only search in particular directories, please specify them in the target_directories field.\nUnless there is a clear reason to use your own search query, please just reuse the user's exact query with their wording.\nTheir exact wording/phrasing can often be helpful for the semantic search query. Keeping the same exact question format can also be helpful.\nThis should be heavily preferred over using the grep search, file search, and list dir tools.", "name": "codebase_search", "parameters": {"properties": {"explanation": {"description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.", "type": "string"}, "query": {"description": "The search query to find relevant code. You should reuse the user's exact query/most recent message with their wording unless there is a clear reason not to.", "type": "string"}, "target_directories": {"description": "Glob patterns for directories to search over", "items": {"type": "string"}, "type": "array"}}, "required": ["query"], "type": "object"}}

{"description": "Read the contents of a file. The output of this tool call will be the 1-indexed file contents from offset to offset+limit (inclusive), together with a summary of the lines outside offset and offset+limit.\nNote that this call can view at most 750 lines at a time and 150 lines minimum.\n\nEach time you call this command you should:\n1) Assess if the contents you viewed are sufficient to proceed with your task.\n2) If you need to read multiple parts of the file, prefer to call the tool once with a larger line range.\n3) If you have found the place to edit or a reasonable answer, do not continue calling tools.\n\nIn some cases, if reading a range of lines is not enough, you may choose to read the entire file.\n\nEither provide the offset and limit, or the should_read_entire_file flag with true.", "name": "read_file", "parameters": {"properties": {"limit": {"description": "The number of lines to read.", "type": "integer"}, "offset": {"description": "The offset to start reading from.", "type": "integer"}, "should_read_entire_file": {"description": "Whether to read the entire file.", "type": "boolean"}, "target_file": {"description": "The path of the file to read. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is.", "type": "string"}}, "required": ["target_file"], "type": "object"}}

{"description": "PROPOSE a command to run on behalf of the user.\nIf you have this tool, note that you DO have the ability to run commands directly on the USER's system.\nNote that the user will have to approve the command before it is executed.\nThe user may reject it if it is not to their liking, or may modify the command before approving it.  If they do change it, take those changes into account.\nThe actual command will NOT execute until the user approves it. The user may not approve it immediately. Do NOT assume the command has started running.\nIf the step is WAITING for user approval, it has NOT started running.\nIn using these tools, adhere to the following guidelines:\n1. Based on the contents of the conversation, you will be told if you are in the same shell as a previous step or a different shell.\n2. If in a new shell, you should `cd` to the appropriate directory and do necessary setup in addition to running the command.\n3. If in the same shell, the state will persist (eg. if you cd in one step, that cwd is persisted next time you invoke this tool).\n4. For ANY commands that would use a pager or require user interaction, you should append ` | cat` to the command (or whatever is appropriate). Otherwise, the command will break. You MUST do this for: git, less, head, tail, more, etc.\n5. For commands that are long running/expected to run indefinitely until interruption, please run them in the background. To run jobs in the background, set `is_background` to true rather than changing the details of the command.\n6. Dont include any newlines in the command.", "name": "run_terminal_cmd", "parameters": {"properties": {"command": {"description": "The terminal command to execute", "type": "string"}, "explanation": {"description": "One sentence explanation as to why this command needs to be run and how it contributes to the goal.", "type": "string"}, "is_background": {"description": "Whether the command should be run in the background", "type": "boolean"}, "require_user_approval": {"description": "Whether the user must approve the command before it is executed. Only set this to false if the command is safe and if it matches the user's requirements for commands that should be executed automatically.", "type": "boolean"}}, "required": ["command", "is_background", "require_user_approval"], "type": "object"}}

{"description": "List the contents of a directory. The quick tool to use for discovery, before using more targeted tools like semantic search or file reading. Useful to try to understand the file structure before diving deeper into specific files. Can be used to explore the codebase.", "name": "list_dir", "parameters": {"properties": {"explanation": {"description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.", "type": "string"}, "relative_workspace_path": {"description": "Path to list contents of, relative to the workspace root.", "type": "string"}}, "required": ["relative_workspace_path"], "type": "object"}}

{"description": "Fast text-based regex search that finds exact pattern matches within files or directories, utilizing the ripgrep command for efficient searching.\nResults will be formatted in the style of ripgrep and can be configured to include line numbers and content.\nTo avoid overwhelming output, the results are capped at 50 matches.\nUse the include or exclude patterns to filter the search scope by file type or specific paths.\n\nThis is best for finding exact text matches or regex patterns.\nMore precise than semantic search for finding specific strings or patterns.\nThis is preferred over semantic search when we know the exact symbol/function name/etc. to search in some set of directories/file types.", "name": "grep_search", "parameters": {"properties": {"case_sensitive": {"description": "Whether the search should be case sensitive", "type": "boolean"}, "exclude_pattern": {"description": "Glob pattern for files to exclude", "type": "string"}, "explanation": {"description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.", "type": "string"}, "include_pattern": {"description": "Glob pattern for files to include (e.g. '*.ts' for TypeScript files)", "type": "string"}, "query": {"description": "The regex pattern to search for", "type": "string"}}, "required": ["query"], "type": "object"}}

{"description": "Use this tool to propose an edit to an existing file.\n\nThis will be read by a less intelligent model, which will quickly apply the edit. You should make it clear what the edit is, while also minimizing the unchanged code you write.\nWhen writing the edit, you should specify each edit in sequence, with the special comment `// ... existing code ...` to represent unchanged code in between edited lines.\n\nFor example:\n\n```\n// ... existing code ...\nFIRST_EDIT\n// ... existing code ...\nSECOND_EDIT\n// ... existing code ...\nTHIRD_EDIT\n// ... existing code ...\n```\n\nYou should still bias towards repeating as few lines of the original file as possible to convey the change.\nBut, each edit should contain sufficient context of unchanged lines around the code you're editing to resolve ambiguity.\nDO NOT omit spans of pre-existing code (or comments) without using the `// ... existing code ...` comment to indicate its absence. If you omit the existing code comment, the model may inadvertently delete these lines.\nMake sure it is clear what the edit should be, and where it should be applied.\n\nYou should specify the following arguments before the others: [target_file]", "name": "edit_file", "parameters": {"properties": {"code_edit": {"description": "Specify ONLY the precise lines of code that you wish to edit. **NEVER specify or write out unchanged code**. Instead, represent all unchanged code using the comment of the language you're editing in - example: `// ... existing code ...`", "type": "string"}, "instructions": {"description": "A single sentence instruction describing what you are going to do for the sketched edit. This is used to assist the less intelligent model in applying the edit. Please use the first person to describe what you are going to do. Dont repeat what you have said previously in normal messages. And use it to disambiguate uncertainty in the edit.", "type": "string"}, "target_file": {"description": "The target file to modify. Always specify the target file as the first argument. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is.", "type": "string"}}, "required": ["target_file", "instructions", "code_edit"], "type": "object"}}

{"description": "Fast file search based on fuzzy matching against file path. Use if you know part of the file path but don't know where it's located exactly. Response will be capped to 10 results. Make your query more specific if need to filter results further.", "name": "file_search", "parameters": {"properties": {"explanation": {"description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.", "type": "string"}, "query": {"description": "Fuzzy filename to search for", "type": "string"}}, "required": ["query", "explanation"], "type": "object"}}

{"description": "Deletes a file at the specified path. The operation will fail gracefully if:\n    - The file doesn't exist\n    - The operation is rejected for security reasons\n    - The file cannot be deleted", "name": "delete_file", "parameters": {"properties": {"explanation": {"description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.", "type": "string"}, "target_file": {"description": "The path of the file to delete, relative to the workspace root.", "type": "string"}}, "required": ["target_file"], "type": "object"}}

{"description": "Calls a smarter model to apply the last edit to the specified file.\nUse this tool immediately after the result of an edit_file tool call ONLY IF the diff is not what you expected, indicating the model applying the changes was not smart enough to follow your instructions.", "name": "reapply", "parameters": {"properties": {"target_file": {"description": "The relative path to the file to reapply the last edit to. You can use either a relative path in the workspace or an absolute path. If an absolute path is provided, it will be preserved as is.", "type": "string"}}, "required": ["target_file"], "type": "object"}}

{"description": "Search the web for real-time information about any topic. Use this tool when you need up-to-date information that might not be available in your training data, or when you need to verify current facts. The search results will include relevant snippets and URLs from web pages. This is particularly useful for questions about current events, technology updates, or any topic that requires recent information.", "name": "web_search", "parameters": {"properties": {"explanation": {"description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.", "type": "string"}, "search_term": {"description": "The search term to look up on the web. Be specific and include relevant keywords for better results. For technical queries, include version numbers or dates if relevant.", "type": "string"}}, "required": ["search_term"], "type": "object"}}

{"description": "Retrieve the history of recent changes made to files in the workspace. This tool helps understand what modifications were made recently, providing information about which files were changed, when they were changed, and how many lines were added or removed. Use this tool when you need context about recent modifications to the codebase.", "name": "diff_history", "parameters": {"properties": {"explanation": {"description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.", "type": "string"}}, "required": [], "type": "object"}}

{"description": "Search for symbols (functions, classes, variables, etc.) in the workspace that match the query. This is a semantic search that uses VSCode's symbol search functionality.\n\nThis tool is useful when you want to:\n1. Find all occurrences of a specific function, class, or variable\n2. Locate where a symbol is defined or used\n3. Get an overview of available symbols in the workspace\n\nThe search is semantic, meaning it understands the structure of the code and can find symbols even if they're not exact text matches.", "name": "search_symbols", "parameters": {"properties": {"query": {"description": "The search query to find symbols. This can be a partial name, and the search will find all matching symbols.", "type": "string"}}, "required": ["query"], "type": "object"}}

Answer the user's request using the relevant tool(s), if they are available. Check that all the required parameters for each tool call are provided or can reasonably be inferred from context. IF there are no relevant tools or there are missing values for required parameters, ask the user to supply these values; otherwise proceed with the tool calls. If the user provides a specific value for a parameter (for example provided in quotes), make sure to use that value EXACTLY. DO NOT make up values for or ask about optional parameters. Carefully analyze descriptive terms in the request as they may indicate required parameter values that should be included even if not explicitly quoted.
"""

# Initialize a cache to track recent code edits (key: hash of edit, value: count)
# TTL of 300 seconds (5 minutes) should be enough to prevent recursive edits in a single conversation
code_edit_cache = TTLCache(maxsize=100, ttl=300)

# Track consecutive edits to the same file
file_edit_counter = TTLCache(maxsize=50, ttl=600)  # 10 minutes TTL
MAX_CONSECUTIVE_EDITS = 3  # Maximum number of consecutive edits to the same file

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    # Add CORS headers to every response
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Expose-Headers', 'X-Request-ID, openai-organization, openai-processing-ms, openai-version')
    response.headers.add('Access-Control-Max-Age', '86400')  # 24 hours
    
    # Only log response status and minimal headers for reduced verbosity
    if response.status_code != 200:
        logger.info(f"Response status: {response.status}")
    
    # Log raw response data only for non-streaming responses
    try:
        content_type = response.headers.get('Content-Type', '')
        if 'text/event-stream' not in content_type:
            response_data = response.get_data(as_text=True)
            # Only log if it's not too large
            if len(response_data) < 5000:
                log_raw_data(f"RESPONSE (Status: {response.status_code})", response_data)
            else:
                # Just log a summary for large responses
                log_raw_data(f"RESPONSE (Status: {response.status_code})", 
                            f"Large response ({len(response_data)} bytes) with content type: {content_type}")
    except Exception as e:
        raw_logger.error(f"Error logging response: {str(e)}")
    
    return response

@app.route('/debug', methods=['GET'])
def debug():
    """Return debug information"""
    return jsonify({
        "status": "running",
        "endpoints": [
            "/v1/chat/completions",
            "/chat/completions",
            "/<path>/chat/completions",
            "/direct",
            "/simple",
            "/agent"
        ],
        "models": list(MODEL_MAPPING.keys()),
        "groq_api_key_set": bool(GROQ_API_KEY),
        "agent_mode_enabled": AGENT_MODE_ENABLED,
        "custom_models": {
            "r1sonqwen": {
                "description": "A chain that uses Deepseek R1 for reasoning and Qwen for code generation",
                "base_models": ["deepseek-r1-distill-qwen-32b", "qwen-2.5-coder-32b"],
                "cache_size": len(r1_reasoning_cache),
                "cache_ttl": "30 minutes"
            }
        }
    })

def format_openai_response(groq_response, original_model):
    """Format Groq response with minimal transformation"""
    try:
        # Start with the original response
        response = groq_response.copy()
        
        # Ensure only required fields exist and have correct format
        if "choices" in response and len(response["choices"]) > 0:
            for choice in response["choices"]:
                if "message" in choice and "role" not in choice["message"]:
                    choice["message"]["role"] = "assistant"
        
        # Ensure basic required fields exist
        if "created" not in response:
            response["created"] = int(time.time())
        if "object" not in response:
            response["object"] = "chat.completion"
        
        logger.info(f"Passing through Groq response with minimal transformation")
        return response
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a basic response structure
        return {
            "object": "chat.completion",
            "created": int(time.time()),
            "model": original_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi!"
                    },
                    "finish_reason": "stop"
                }
            ]
        }

# Handle OPTIONS requests for all routes
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle OPTIONS requests for all routes"""
    logger.info(f"OPTIONS request: /{path}")
    
    # Create a response with all the necessary CORS headers
    response = make_response('')
    response.status_code = 200
    
    # Add all the headers that Cursor might expect
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin')
    response.headers.add('Access-Control-Expose-Headers', 'X-Request-ID, openai-organization, openai-processing-ms, openai-version')
    response.headers.add('Access-Control-Max-Age', '86400')  # 24 hours
    
    return response

def process_chat_request():
    """Process a chat completion request from any endpoint"""
    try:
        # Get client IP (for logging purposes)
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Log the request info but less verbosely
        logger.info(f"Request from {client_ip} using {user_agent.split(' ')[0]}")
        
        # Log raw request data
        log_raw_data("REQUEST HEADERS", dict(request.headers))
        
        if request.is_json:
            log_raw_data("REQUEST JSON BODY", request.json)
        else:
            log_raw_data("REQUEST RAW BODY", request.data.decode('utf-8', errors='replace'))
        
        # Handle preflight OPTIONS request
        if request.method == 'OPTIONS':
            logger.info("OPTIONS preflight request")
            return handle_options(request.path.lstrip('/'))
        
        # Get the request data
        if request.is_json:
            data = request.json
            # Log message count and types without full content
            if 'messages' in data:
                messages = data['messages']
                msg_summary = [f"{m.get('role', 'unknown')}: {len(m.get('content', ''))}" for m in messages]
                logger.info(f"Processing {len(messages)} messages: {msg_summary}")
                
                # Take only the last few messages if there are too many
                if len(messages) > 10:
                    logger.info(f"Truncating message history from {len(messages)} to last 10 messages")
                    # Always include the system message if present
                    system_messages = [m for m in messages if m.get('role') == 'system']
                    other_messages = [m for m in messages if m.get('role') != 'system']
                    
                    # Keep system messages and last 9 other messages
                    truncated_messages = system_messages + other_messages[-9:]
                    data['messages'] = truncated_messages
                    logger.info(f"Truncated to {len(truncated_messages)} messages")
            
            # Log model information
            if 'model' in data:
                model = data['model']
                logger.info(f"Request for model: {model}")
                
                # Check if the model is r1sonqwen
                if model == 'r1sonqwen':
                    logger.info("Processing r1sonqwen request")
                    return process_r1sonqwen_request(data)
                
                # Map to Groq model if needed
                if model in MODEL_MAPPING:
                    groq_model = MODEL_MAPPING[model]
                else:
                    groq_model = MODEL_MAPPING["default"]
            else:
                groq_model = MODEL_MAPPING["default"]
                logger.info(f"No model specified, using default: {groq_model}")
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
                logger.info(f"Non-JSON request parsed for model: {data.get('model', 'unknown')}")
            except:
                logger.error("Failed to parse request data")
                data = {}
        
        # Check cache for this exact request
        cache_key = None
        if request.is_json:
            try:
                cache_key = json.dumps(data, sort_keys=True)
                if cache_key in request_cache:
                    logger.info("Using cached response for duplicate request")
                    return request_cache[cache_key]
            except Exception as e:
                logger.error(f"Error checking cache: {str(e)}")
        
        # Always enable streaming for better reliability
        request_data = data.copy()
        request_data['stream'] = True
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending streaming request to Groq with {len(request_data.get('messages', []))} messages")
        log_raw_data("GROQ REQUEST", request_data)
        
        def generate():
            try:
                # Create a list to collect streaming chunks for logging
                collected_chunks = []
                
                with requests.post(
                    f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                    json=request_data,
                    headers=headers,
                    stream=True,
                    timeout=GROQ_TIMEOUT
                ) as groq_response:
                    
                    # Check for error status
                    if groq_response.status_code != 200:
                        error_msg = groq_response.text[:200] if hasattr(groq_response, 'text') else "Unknown error"
                        logger.error(f"Groq API error: {groq_response.status_code} - {error_msg}")
                        error_response = {
                            "error": {
                                "message": f"Groq API error: {groq_response.status_code}",
                                "type": "server_error",
                                "code": "groq_error"
                            }
                        }
                        log_raw_data("GROQ ERROR RESPONSE", error_response)
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Process the streaming response
                    for line in groq_response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            # Collect the chunk for logging instead of logging each one
                            collected_chunks.append(line)
                            
                            if line.startswith('data: '):
                                # Pass through the streaming data
                                yield f"{line}\n\n"
                            elif line.strip() == 'data: [DONE]':
                                yield "data: [DONE]\n\n"
                
                # Log all collected chunks at once
                if collected_chunks:
                    log_raw_data("GROQ STREAMING RESPONSE (COMPLETE)", 
                                 collect_streaming_chunks(collected_chunks))

            except requests.exceptions.Timeout:
                logger.error("Groq API timeout")
                error_response = {
                    "error": {
                        "message": "Request timeout",
                        "type": "timeout_error",
                        "code": "timeout"
                    }
                }
                log_raw_data("TIMEOUT ERROR", error_response)
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "stream_error"
                    }
                }
                log_raw_data("STREAMING ERROR", {"error": str(e), "traceback": traceback.format_exc()})
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Clear the cache after processing the request
                if cache_key in request_cache:
                    del request_cache[cache_key]
                    logger.info("Cache cleared for request")

        # Return a streaming response
        response = app.response_class(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'access-control-expose-headers': 'X-Request-ID',
                'x-request-id': str(uuid.uuid4())
            }
        )
        
        logger.info("Started streaming response")
        return response
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a properly structured error response
        error_response_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": "no_completion"
            }
        }
        
        error_response = make_response(jsonify(error_response_data))
        error_response.status_code = 500
        error_response.headers.add('Content-Type', 'application/json')
        
        return error_response

# Route for standard OpenAI endpoint
@app.route(OPENAI_CHAT_ENDPOINT, methods=['POST', 'OPTIONS'])
def openai_chat_completions():
    logger.info(f"Request to standard OpenAI endpoint")
    return process_chat_request()

# Route for Cursor's custom endpoint
@app.route(CURSOR_CHAT_ENDPOINT, methods=['POST', 'OPTIONS'])
def cursor_chat_completions():
    logger.info(f"Request to Cursor endpoint")
    return process_chat_request()

# Catch-all route for any other chat completions endpoint
@app.route('/<path:path>/chat/completions', methods=['POST', 'OPTIONS'])
def any_chat_completions(path):
    logger.info(f"Request to custom path: /{path}/chat/completions")
    return process_chat_request()

# Add a route for OpenAI's models endpoint
@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    """Return a fake list of models"""
    logger.info("Request to models endpoint")
    
    models = [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        },
        {
            "id": "gpt-4o-2024-08-06",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        },
        {
            "id": "default",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        }
    ]
    
    # Create response with OpenAI-specific headers
    response = make_response(jsonify({"data": models, "object": "list"}))
    
    # Add OpenAI specific headers (avoiding hop-by-hop headers)
    response.headers.add('access-control-expose-headers', 'X-Request-ID')
    response.headers.add('openai-organization', 'user-68tm5q5hm9sro0tao3xi5e9i')
    response.headers.add('openai-processing-ms', '10')
    response.headers.add('openai-version', '2020-10-01')
    response.headers.add('strict-transport-security', 'max-age=15724800; includeSubDomains')
    response.headers.add('x-request-id', str(uuid.uuid4()))
    
    # Set correct Content-Type header
    response.headers.set('Content-Type', 'application/json')
    
    return response

# Add health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Return health status of the proxy server"""
    logger.info("Health check request")
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - start_time
    })

@app.route('/', methods=['GET'])
def home():
    logger.info("Home page request")
    return """
    <html>
    <head>
        <title>Groq Proxy Server</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
            .endpoint { background-color: #e0f7fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Groq Proxy Server is running</h1>
        <p>This server proxies requests to the following endpoints:</p>
        <div class="endpoint">
            <h3>/v1/chat/completions (Standard OpenAI endpoint)</h3>
            <p>Use this endpoint for standard OpenAI API compatibility</p>
        </div>
        <div class="endpoint">
            <h3>/chat/completions (Cursor endpoint)</h3>
            <p>Use this endpoint for Cursor compatibility</p>
        </div>
        <div class="endpoint">
            <h3>/direct (Direct endpoint)</h3>
            <p>Simple endpoint that takes a single message and returns a response</p>
        </div>
        <div class="endpoint">
            <h3>/simple (Simple endpoint)</h3>
            <p>Simple non-streaming endpoint with OpenAI-compatible response format</p>
        </div>
        <div class="endpoint">
            <h3>/agent (Agent mode endpoint)</h3>
            <p>Endpoint with agent mode instructions included in system prompt</p>
        </div>
        
        <h2>Test the API</h2>
        <p>You can test the API with the following curl command:</p>
        <pre>
curl -X POST \\
  [YOUR_NGROK_URL]/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer fake-api-key" \\
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a test assistant."},
      {"role": "user", "content": "Testing. Just say hi and nothing else."}
    ]
  }'
        </pre>
        
        <h2>Test Agent Mode</h2>
        <p>You can test the agent mode with the following curl command:</p>
        <pre>
curl -X POST \\
  [YOUR_NGROK_URL]/agent \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer fake-api-key" \\
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Please help me understand how to use the agent mode."}
    ]
  }'
        </pre>
        
        <h2>Debug Information</h2>
        <p>For debug information, visit <a href="/debug">/debug</a></p>
        <p>For health check, visit <a href="/health">/health</a></p>
    </body>
    </html>
    """

def start_ngrok(port):
    """Start ngrok and return the public URL"""
    try:
        # Check if ngrok is installed
        try:
            subprocess.run(["ngrok", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ngrok is not installed or not in PATH. Please install ngrok first.")
            print("ngrok is not installed or not in PATH. Please install ngrok first.")
            print("Visit https://ngrok.com/download to download and install ngrok")
            sys.exit(1)
            
        # Start ngrok with recommended settings for Cursor
        logger.info(f"Starting ngrok on port {port}...")
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Started ngrok process (PID: {ngrok_process.pid})")
        
        # Wait for ngrok to start
        logger.info("Waiting for ngrok to initialize...")
        time.sleep(3)
        
        # Get the public URL from ngrok API
        try:
            logger.info("Requesting tunnel information from ngrok API...")
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()["tunnels"]
            if tunnels:
                # Using https tunnel is recommended for Cursor
                https_tunnels = [t for t in tunnels if t["public_url"].startswith("https")]
                if https_tunnels:
                    public_url = https_tunnels[0]["public_url"]
                else:
                    public_url = tunnels[0]["public_url"]
                
                logger.info(f"ngrok public URL: {public_url}")
                
                print(f"\n{'='*60}")
                print(f"NGROK PUBLIC URL: {public_url}")
                print(f"NGROK INSPECTOR: http://localhost:4040")
                print(f"Use this URL in Cursor as your OpenAI API base URL")
                print(f"{'='*60}\n")
                
                # Print example PowerShell command
                print("Example PowerShell command to test the proxy:")
                print(f"""
$headers = @{{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer fake-api-key"
}}

$body = @{{
    "messages" = @(
        @{{
            "role" = "system"
            "content" = "You are a test assistant."
        }},
        @{{
            "role" = "user"
            "content" = "Testing. Just say hi and nothing else."
        }}
    )
    "model" = "gpt-4o"
}} | ConvertTo-Json

Invoke-WebRequest -Uri "{public_url}/v1/chat/completions" -Method Post -Headers $headers -Body $body
                """)
                
                # Print curl command for testing
                print("\nOr use this curl command:")
                print(f"""
curl -X POST \\
  {public_url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer fake-api-key" \\
  -d '{{
    "model": "gpt-4o",
    "messages": [
      {{"role": "system", "content": "You are a test assistant."}},
      {{"role": "user", "content": "Testing. Just say hi and nothing else."}}
    ]
  }}'
                """)
                
                # Print instructions for Cursor
                print("\nTo configure Cursor:")
                print(f"1. Set the OpenAI API base URL to: {public_url}")
                print("2. Use any OpenAI model name that Cursor supports")
                print("3. Set any API key (it won't be checked)")
                print("4. Check the ngrok inspector at http://localhost:4040 to debug traffic")
                
                return public_url
            else:
                logger.error("No ngrok tunnels found")
                print("No ngrok tunnels found. Please check ngrok configuration.")
                return None
        except Exception as e:
            logger.error(f"Error getting ngrok URL: {str(e)}")
            print(f"Error getting ngrok URL: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error starting ngrok: {str(e)}")
        print(f"Error starting ngrok: {str(e)}")
        return None

# Store app start time for uptime tracking
start_time = time.time()

# Add a new direct endpoint for simple message passing
@app.route('/direct', methods=['POST', 'OPTIONS'])
def direct_completion():
    """Simple endpoint that takes a single message and returns a response"""
    logger.info("Request to direct endpoint")
    
    if request.method == 'OPTIONS':
        return handle_options('direct')
    
    try:
        # Get the request data
        if request.is_json:
            data = request.json
            message = data.get('message', '')
            model = data.get('model', 'qwen-2.5-coder-32b')
            logger.info(f"Direct request for model: {model}")
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
                message = data.get('message', '')
                model = data.get('model', 'qwen-2.5-coder-32b')
            except:
                logger.error("Failed to parse direct request data")
                return jsonify({"error": "Invalid request format"}), 400
        
        # Create a simple request to Groq
        groq_request = {
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ],
            "stream": False  # No streaming for direct endpoint
        }
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending direct request to Groq")
        log_raw_data("DIRECT REQUEST", groq_request)
        
        response = requests.post(
            f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
            json=groq_request,
            headers=headers,
            timeout=GROQ_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text[:200]}")
            log_raw_data("DIRECT ERROR RESPONSE", response.text)
            return jsonify({
                "error": f"Groq API error: {response.status_code}",
                "message": "Failed to get response from Groq"
            }), response.status_code
        
        # Parse the response
        log_raw_data("DIRECT RAW RESPONSE", response.text)
        groq_response = response.json()
        log_raw_data("DIRECT PARSED RESPONSE", groq_response)
        
        # Extract just the content from the response
        if "choices" in groq_response and len(groq_response["choices"]) > 0:
            content = groq_response["choices"][0]["message"]["content"]
            result = {"response": content}
            log_raw_data("DIRECT FINAL RESPONSE", result)
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error processing direct request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Add a simple non-streaming endpoint for Cursor
@app.route('/simple', methods=['POST', 'OPTIONS'])
def simple_completion():
    """Simple non-streaming endpoint for Cursor"""
    logger.info("Request to simple endpoint")
    
    if request.method == 'OPTIONS':
        return handle_options('simple')
    
    try:
        # Get client IP (for logging purposes)
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Log the request info
        logger.info(f"Simple request from {client_ip} using {user_agent.split(' ')[0]}")
        
        # Get the request data
        if request.is_json:
            data = request.json
            # Log message count without full content
            if 'messages' in data:
                messages = data['messages']
                msg_count = len(messages)
                logger.info(f"Processing {msg_count} messages in simple mode")
                
                # Take only the last few messages if there are too many
                if len(messages) > 10:
                    logger.info(f"Truncating message history from {len(messages)} to last 10 messages")
                    # Always include the system message if present
                    system_messages = [m for m in messages if m.get('role') == 'system']
                    other_messages = [m for m in messages if m.get('role') != 'system']
                    
                    # Keep system messages and last 9 other messages
                    truncated_messages = system_messages + other_messages[-9:]
                    data['messages'] = truncated_messages
                    logger.info(f"Truncated to {len(truncated_messages)} messages")
            
            # Log model information
            if 'model' in data:
                model = data['model']
                logger.info(f"Simple request for model: {model}")
                # Map to Groq model if needed
                if model in MODEL_MAPPING:
                    groq_model = MODEL_MAPPING[model]
                else:
                    groq_model = MODEL_MAPPING["default"]
            else:
                groq_model = MODEL_MAPPING["default"]
                logger.info(f"No model specified, using default: {groq_model}")
        else:
            logger.error("Failed to parse request data")
            return jsonify({"error": "Invalid request format"}), 400
        
        # Create request for Groq
        groq_request = data.copy()
        groq_request['model'] = groq_model
        groq_request['stream'] = False  # Explicitly disable streaming
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending non-streaming request to Groq")
        log_raw_data("SIMPLE REQUEST", groq_request)
        
        response = requests.post(
            f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
            json=groq_request,
            headers=headers,
            timeout=GROQ_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text[:200]}")
            log_raw_data("SIMPLE ERROR RESPONSE", response.text)
            return jsonify({
                "error": {
                    "message": f"Groq API error: {response.status_code}",
                    "type": "server_error",
                    "code": "groq_error"
                }
            }), response.status_code
        
        # Parse the response
        log_raw_data("SIMPLE RAW RESPONSE", response.text)
        groq_response = response.json()
        log_raw_data("SIMPLE PARSED RESPONSE", groq_response)
        
        # Format as OpenAI response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,  # Use the original model name
            "choices": groq_response.get("choices", []),
            "usage": groq_response.get("usage", {})
        }
        
        log_raw_data("SIMPLE FORMATTED RESPONSE", openai_response)
        logger.info(f"Successfully processed simple request")
        return jsonify(openai_response)
            
    except Exception as e:
        logger.error(f"Error processing simple request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a properly structured error response
        error_response_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": "no_completion"
            }
        }
        
        return jsonify(error_response_data), 500

def handle_model_instructions(model_name, instructions):
    """
    This function demonstrates how to handle model instructions and tool usage.
    
    Parameters:
    model_name (str): The name of the model (e.g., 'qwen-2.5-coder-32b')
    instructions (str): The instructions for the model
    
    Returns:
    dict: A dictionary containing the response
    """
    logger.info(f"Handling instructions for model: {model_name}")
    
    # Process the instructions
    response = {
        "model": model_name,
        "instructions": instructions,
        "processed": True,
        "timestamp": time.time()
    }
    
    # Log the response
    logger.info(f"Processed instructions for model: {model_name}")
    
    return response

def handle_tool_usage(tool_name, parameters):
    """
    This function demonstrates how to handle tool usage for models.
    
    Parameters:
    tool_name (str): The name of the tool to use
    parameters (dict): The parameters for the tool
    
    Returns:
    dict: A dictionary containing the response
    """
    logger.info(f"Using tool: {tool_name} with parameters: {parameters}")
    
    # Check for recursive code edits if this is an edit_file tool
    if tool_name == "edit_file" and "code_edit" in parameters:
        # Create a simple hash of the edit to use as a cache key
        edit_hash = hash(parameters.get("code_edit", ""))
        target_file = parameters.get("target_file", "")
        cache_key = f"{target_file}:{edit_hash}"
        
        # Check if we've seen this edit before
        if cache_key in code_edit_cache:
            # Increment the count
            code_edit_cache[cache_key] += 1
            count = code_edit_cache[cache_key]
            
            # If we've seen this edit more than twice, return an error
            if count > 2:
                logger.warning(f"Detected recursive code edit attempt ({count} times) for {target_file}")
                return {
                    "error": True,
                    "message": "Recursive code edit detected. Please try a different approach or ask the user for guidance.",
                    "tool": tool_name,
                    "parameters": parameters,
                    "timestamp": time.time()
                }
        else:
            # First time seeing this edit
            code_edit_cache[cache_key] = 1
        
        # Track consecutive edits to the same file
        if target_file in file_edit_counter:
            file_edit_counter[target_file] += 1
        else:
            file_edit_counter[target_file] = 1
        
        # Check if we've exceeded the maximum number of consecutive edits
        if file_edit_counter[target_file] > MAX_CONSECUTIVE_EDITS:
            logger.warning(f"Exceeded maximum consecutive edits ({MAX_CONSECUTIVE_EDITS}) for {target_file}")
            return {
                "error": True,
                "message": f"You've made {file_edit_counter[target_file]} consecutive edits to {target_file}. Please take a step back and reconsider your approach or ask the user for guidance.",
                "tool": tool_name,
                "parameters": parameters,
                "timestamp": time.time()
            }
    elif tool_name != "edit_file":
        # Reset the consecutive edit counter for all files if we're using a different tool
        file_edit_counter.clear()
    
    # Process the tool usage
    response = {
        "tool": tool_name,
        "parameters": parameters,
        "processed": True,
        "timestamp": time.time()
    }
    
    # Log the response
    logger.info(f"Processed tool usage: {tool_name}")
    
    return response

def send_request_to_groq(request_data):
    """
    Send a request to Groq API and return the response
    
    Parameters:
    request_data (dict): The request data to send to Groq
    
    Returns:
    dict: The response from Groq
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    # Log the request
    logger.info(f"Sending request to Groq for model: {request_data.get('model', 'unknown')}")
    
    # Try up to MAX_RETRIES times
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                json=request_data,
                headers=headers,
                timeout=GROQ_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Groq API error (attempt {attempt+1}/{MAX_RETRIES}): {response.status_code} - {response.text[:200]}")
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Groq API error: {response.status_code} - {response.text[:200]}")
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Error sending request to Groq (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # This should never be reached due to the exception in the loop
    raise Exception("Failed to send request to Groq after multiple attempts")

def extract_content_from_response(response):
    """
    Extract the content from a Groq API response
    
    Parameters:
    response (dict): The response from Groq
    
    Returns:
    str: The content from the response
    """
    if not response.get('choices'):
        raise Exception("No choices in response")
    
    return response['choices'][0]['message']['content']

def process_r1sonqwen_request(data):
    """
    Process a request using the R1sonQwen chain:
    1. Use R1 to create a reasoning chain based on Cursor system prompts
    2. Pass the reasoning and original system prompts to Qwen
    3. Return Qwen's response with minimal transformation to match Cursor expectations
    """
    try:
        logger.info("Starting r1sonqwen chain processing")
        
        # Extract the original request and system prompts
        original_messages = data.get('messages', [])
        logger.info(f"Original request has {len(original_messages)} messages")
        
        # Check if streaming is requested
        stream_mode = data.get('stream', False)
        logger.info(f"Stream mode: {stream_mode}")
        
        # Try to get reasoning from R1 or cache
        r1_reasoning = None
        try:
            cache_key = json.dumps(data, sort_keys=True)
            if cache_key in r1_reasoning_cache:
                logger.info("Using cached R1 reasoning")
                r1_reasoning = r1_reasoning_cache[cache_key]
                logger.info(f"Retrieved cached reasoning of length: {len(r1_reasoning)}")
            else:
                logger.info("No cached reasoning found, proceeding with R1 call")
                
                # Create R1 request with focus on reasoning chain
                r1_request = {
                    "model": "deepseek-r1-distill-qwen-32b",
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are a reasoning chain generator. Your task is to analyze the user's request and create a structured reasoning chain that follows this format:

<reasoning_chain>
1. CONTEXT ANALYSIS
- Available files and their purposes
- Current state and issues
- User's specific request

2. IMPLEMENTATION APPROACH
- Required changes
- Potential challenges
- Dependencies and considerations

3. EXECUTION PLAN
- Step-by-step implementation
- Testing requirements
- Success criteria

4. VALIDATION STRATEGY
- Error handling
- Edge cases
- Quality assurance steps
</reasoning_chain>

Focus ONLY on creating this reasoning chain. DO NOT provide any implementation details or code."""
                        }
                    ],
                    "temperature": 0.3,  # Lower temperature for more deterministic reasoning
                    "max_tokens": 1000,
                    "stream": False  # Never stream the R1 request
                }
                
                # Add user messages but filter out assistant messages
                user_messages = [msg for msg in original_messages if msg['role'] in ['user', 'system']]
                r1_request['messages'].extend(user_messages)
                
                # Send request to R1
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}"
                }
                
                log_raw_data("R1 REQUEST", r1_request)
                
                r1_response_raw = requests.post(
                    f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                    json=r1_request,
                    headers=headers,
                    timeout=GROQ_TIMEOUT
                )
                
                logger.info(f"R1 response status: {r1_response_raw.status_code}")
                log_raw_data("R1 RAW RESPONSE", r1_response_raw.text)
                
                if r1_response_raw.status_code == 200:
                    r1_response = r1_response_raw.json()
                    log_raw_data("R1 PARSED RESPONSE", r1_response)
                    
                    if 'choices' in r1_response and len(r1_response['choices']) > 0:
                        r1_reasoning = r1_response['choices'][0]['message']['content']
                        logger.info(f"Successfully extracted reasoning chain (length: {len(r1_reasoning)})")
                        r1_reasoning_cache[cache_key] = r1_reasoning
        except Exception as e:
            logger.error(f"Error getting reasoning from R1: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue without reasoning
            r1_reasoning = None
        
        # Create Qwen request
        qwen_request = {
            "model": "qwen-2.5-coder-32b",
            "messages": original_messages.copy(),
            "temperature": data.get('temperature', 0.7),
            "max_tokens": data.get('max_tokens', 1000),
            "stream": stream_mode
        }
        
        # Add the reasoning as a system message if there isn't already one and if we have reasoning
        if r1_reasoning:
            has_system = False
            for msg in qwen_request["messages"]:
                if msg.get("role") == "system":
                    has_system = True
                    # Append reasoning to existing system message
                    msg["content"] += f"\n\nReasoning chain:\n{r1_reasoning}"
                    break
            
            if not has_system:
                # Insert a system message with the reasoning at the beginning
                qwen_request["messages"].insert(0, {
                    "role": "system",
                    "content": f"Reasoning chain:\n{r1_reasoning}"
                })
        
        # Forward to Qwen
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending request to Qwen with stream={stream_mode}")
        log_raw_data("QWEN REQUEST", qwen_request)
        
        if stream_mode:
            # Handle streaming response
            return handle_qwen_streaming(qwen_request, headers)
        else:
            # Handle non-streaming response
            return handle_qwen_non_streaming(qwen_request, headers)
    
    except Exception as e:
        logger.error(f"Error in R1sonQwen chain: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response in the same format as OpenAI
        error_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "r1sonqwen",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error in R1sonQwen chain: {str(e)}"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        return jsonify(error_response)

# Helper function for Qwen streaming
def handle_qwen_streaming(qwen_request, headers):
    """Handle streaming response from Qwen - with special handling for code blocks"""
    def generate():
        try:
            # Create a list to collect streaming chunks for logging
            collected_chunks = []
            
            # Track if we're in a code block to prevent premature closing
            in_code_block = False
            code_block_count = 0
            last_chunk_time = time.time()
            
            with requests.post(
                f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                json=qwen_request,
                headers=headers,
                stream=True,
                timeout=GROQ_TIMEOUT
            ) as groq_response:
                
                # Check for error status
                if groq_response.status_code != 200:
                    error_msg = groq_response.text[:200] if hasattr(groq_response, 'text') else "Unknown error"
                    logger.error(f"Groq API error: {groq_response.status_code} - {error_msg}")
                    error_response = {
                        "error": {
                            "message": f"Groq API error: {groq_response.status_code}",
                            "type": "server_error",
                            "code": "groq_error"
                        }
                    }
                    log_raw_data("QWEN STREAMING ERROR", error_response)
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Process the streaming response
                for line in groq_response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        last_chunk_time = time.time()
                        
                        # Collect the chunk for logging
                        collected_chunks.append(line)
                        
                        # Check if we're entering or exiting a code block
                        if line.startswith('data: ') and '"content":"```' in line:
                            in_code_block = True
                            code_block_count += 1
                            logger.info(f"Entering code block #{code_block_count}")
                        elif line.startswith('data: ') and '"content":"```' in line and in_code_block:
                            in_code_block = False
                            logger.info(f"Exiting code block #{code_block_count}")
                        
                        if line.startswith('data: '):
                            # Only modify the model name, nothing else
                            if '"model":"qwen-2.5-coder-32b"' in line:
                                line = line.replace('"model":"qwen-2.5-coder-32b"', '"model":"r1sonqwen"')
                            # Pass through the streaming data
                            yield f"{line}\n\n"
                        elif line.strip() == 'data: [DONE]':
                            yield "data: [DONE]\n\n"
                
                # Log all collected chunks at once
                if collected_chunks:
                    log_raw_data("QWEN STREAMING RESPONSE (COMPLETE)", 
                                collect_streaming_chunks(collected_chunks))
                
                # If we were in a code block, make sure we send a proper closing
                if in_code_block:
                    logger.info("Detected unclosed code block, sending closing marker")
                    # Send a dummy chunk to keep the connection alive
                    dummy_chunk = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "r1sonqwen",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(dummy_chunk)}\n\n"
                
                # Always send a final [DONE] marker
                yield "data: [DONE]\n\n"
                
                # Wait a moment before closing to ensure all data is processed
                time.sleep(0.5)

        except requests.exceptions.Timeout:
            logger.error("Groq API timeout")
            error_response = {
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error",
                    "code": "timeout"
                }
            }
            log_raw_data("TIMEOUT ERROR", error_response)
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "stream_error"
                }
            }
            log_raw_data("STREAMING ERROR", {"error": str(e), "traceback": traceback.format_exc()})
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"

    # Return a streaming response with keep-alive headers
    response = app.response_class(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'access-control-expose-headers': 'X-Request-ID',
            'x-request-id': str(uuid.uuid4())
        }
    )
    
    logger.info("Started streaming response")
    return response

# Helper function for Qwen non-streaming
def handle_qwen_non_streaming(qwen_request, headers):
    """Handle non-streaming response from Qwen"""
    qwen_response_raw = requests.post(
        f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
        json=qwen_request,
        headers=headers,
        timeout=GROQ_TIMEOUT
    )
    
    if qwen_response_raw.status_code != 200:
        logger.error(f"Qwen API error: {qwen_response_raw.status_code} - {qwen_response_raw.text[:200]}")
        log_raw_data("QWEN ERROR RESPONSE", qwen_response_raw.text)
        raise Exception(f"Qwen API error: {qwen_response_raw.status_code}")
    
    # Log the raw response text for debugging
    logger.info(f"Qwen raw response text (first 500 chars): {qwen_response_raw.text[:500]}...")
    log_raw_data("QWEN RAW RESPONSE", qwen_response_raw.text)
    
    try:
        # Get the raw response and only change the model name
        qwen_response = qwen_response_raw.json()
        qwen_response['model'] = 'r1sonqwen'
        
        log_raw_data("QWEN MODIFIED RESPONSE", qwen_response)
        
        logger.info("Successfully processed r1sonqwen chain")
        logger.info(f"Response structure: {json.dumps(qwen_response)[:500]}...")
        return jsonify(qwen_response)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for Qwen response: {str(e)}")
        logger.error(f"Raw response: {qwen_response_raw.text}")
        raise Exception(f"Failed to parse Qwen response: {str(e)}")

@app.route('/agent', methods=['POST', 'OPTIONS'])
def agent_mode():
    """Special agent mode endpoint that includes agent instructions in the system prompt"""
    logger.info("Request to agent mode endpoint")
    
    if request.method == 'OPTIONS':
        return handle_options('agent')
    
    try:
        # Get client IP (for logging purposes)
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Log the request info
        logger.info(f"Agent request from {client_ip} using {user_agent.split(' ')[0]}")
        
        # Get the request data
        if request.is_json:
            data = request.json
            
            # Check if this is a tool usage request
            if 'tool_call' in data:
                tool_name = data.get('tool_call', {}).get('name')
                parameters = data.get('tool_call', {}).get('parameters', {})
                
                if tool_name:
                    # Process the tool usage
                    tool_response = handle_tool_usage(tool_name, parameters)
                    
                    # Check if there was an error
                    if tool_response.get('error'):
                        logger.warning(f"Tool usage error: {tool_response.get('message')}")
                        
                        # Return a response with the error message
                        error_response = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": data.get('model', 'unknown'),
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"**Error: {tool_response.get('message')}**\n\nPlease try a different approach or ask the user for guidance."
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        }
                        
                        return jsonify(error_response)
            
            # Log message count without full content
            if 'messages' in data:
                messages = data['messages']
                msg_count = len(messages)
                logger.info(f"Processing {msg_count} messages in agent mode")
                
                # Check if the last user message contains a tool call
                last_user_message = None
                for msg in reversed(messages):
                    if msg.get('role') == 'user':
                        last_user_message = msg.get('content', '')
                        break
                
                # If the last user message contains a tool call, check for recursive patterns
                if last_user_message and "edit_file" in last_user_message:
                    # Look for patterns that might indicate recursive behavior
                    recursive_indicators = [
                        "I'll try again",
                        "Let me try again",
                        "Let's try again",
                        "Trying again",
                        "Let me reapply",
                        "I'll reapply"
                    ]
                    
                    if any(indicator in last_user_message for indicator in recursive_indicators):
                        logger.warning("Detected potential recursive behavior in user message")
                        # Add a system message warning about recursive behavior
                        messages.append({
                            "role": "system",
                            "content": "WARNING: Potential recursive behavior detected. Please do not repeatedly attempt the same edit. If an edit is not working, try a different approach or ask the user for guidance."
                        })
                
                # Add or update the system message with agent instructions
                system_message_found = False
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'system':
                        system_message_found = True
                        # Append agent instructions to existing system message
                        if AGENT_INSTRUCTIONS not in msg.get('content', ''):
                            messages[i]['content'] = msg['content'] + "\n\n" + AGENT_INSTRUCTIONS
                
                # If no system message found, add one
                if not system_message_found:
                    messages.insert(0, {
                        "role": "system",
                        "content": AGENT_INSTRUCTIONS
                    })
                    
                # Take only the last few messages if there are too many
                if len(messages) > 10:
                    logger.info(f"Truncating message history from {len(messages)} to last 10 messages")
                    # Always include the system message if present
                    system_messages = [m for m in messages if m.get('role') == 'system']
                    other_messages = [m for m in messages if m.get('role') != 'system']
                    
                    # Keep system messages and last 9 other messages
                    truncated_messages = system_messages + other_messages[-9:]
                    data['messages'] = truncated_messages
                    logger.info(f"Truncated to {len(truncated_messages)} messages")
            
            # Log model information
            if 'model' in data:
                model = data['model']
                logger.info(f"Agent request for model: {model}")
                # Map to Groq model if needed
                if model in MODEL_MAPPING:
                    groq_model = MODEL_MAPPING[model]
                else:
                    groq_model = MODEL_MAPPING["default"]
            else:
                groq_model = MODEL_MAPPING["default"]
                logger.info(f"No model specified, using default: {groq_model}")
        else:
            logger.error("Failed to parse request data")
            return jsonify({"error": "Invalid request format"}), 400
        
        # Create request for Groq
        groq_request = data.copy()
        groq_request['model'] = groq_model
        groq_request['stream'] = False  # Explicitly disable streaming for agent mode
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending agent mode request to Groq")
        log_raw_data("AGENT MODE REQUEST", groq_request)
        
        response = requests.post(
            f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
            json=groq_request,
            headers=headers,
            timeout=GROQ_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text[:200]}")
            log_raw_data("AGENT MODE ERROR RESPONSE", response.text)
            return jsonify({
                "error": {
                    "message": f"Groq API error: {response.status_code}",
                    "type": "server_error",
                    "code": "groq_error"
                }
            }), response.status_code
        
        # Parse the response
        log_raw_data("AGENT MODE RAW RESPONSE", response.text)
        groq_response = response.json()
        log_raw_data("AGENT MODE PARSED RESPONSE", groq_response)
        
        # Check if there's a message about recursive code edits
        if groq_response.get("choices") and len(groq_response["choices"]) > 0:
            content = groq_response["choices"][0].get("message", {}).get("content", "")
            
            # If the response contains indicators of recursive behavior, add a warning
            recursive_indicators = [
                "I'll try again with the edit",
                "Let me try again with the same edit",
                "Let's try the edit again",
                "I'll reapply the same edit"
            ]
            
            if any(indicator in content for indicator in recursive_indicators):
                logger.warning("Detected recursive behavior in model response")
                # Modify the response to include a warning
                warning_message = "\n\n**WARNING: Potential recursive behavior detected. Please try a different approach instead of repeating the same edit.**"
                groq_response["choices"][0]["message"]["content"] = content + warning_message
                log_raw_data("AGENT MODE MODIFIED RESPONSE (with warning)", groq_response)
        
        # Format as OpenAI response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,  # Use the original model name
            "choices": groq_response.get("choices", []),
            "usage": groq_response.get("usage", {})
        }
        
        logger.info(f"Successfully processed agent mode request")
        return jsonify(openai_response)
            
    except Exception as e:
        logger.error(f"Error processing agent mode request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a properly structured error response
        error_response_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": "no_completion"
            }
        }
        
        return jsonify(error_response_data), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Groq proxy server on port {port}")
    
    # Start ngrok in a separate thread
    public_url = start_ngrok(port)
    
    # Start the Flask server
    print(f"Starting Groq proxy server on port {port}")
    logger.info(f"Server starting on port {port}")
    
    try:
        serve(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.critical(f"Server failed to start: {str(e)}")
        print(f"Server failed to start: {str(e)}")
        sys.exit(1) 