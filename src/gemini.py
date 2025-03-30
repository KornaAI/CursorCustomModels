from flask import Flask, request, jsonify, make_response, Response, g
import requests
import os
import json
import logging
from waitress import serve
import time
import sys
import traceback  # Add traceback for error logging
from flask_cors import CORS
import uuid
from dotenv import load_dotenv
import google.generativeai as genai
import subprocess  # Added for ngrok
import datetime  # For timestamping logs

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# API Key configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY is not set in environment variables")
    print("Please set the GOOGLE_API_KEY environment variable and restart the server")

# Configure the Gemini API
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {str(e)}")

# OpenAI API endpoints that we'll intercept
OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"
CURSOR_CHAT_ENDPOINT = "/chat/completions"

# Model mapping - map OpenAI models to Gemini models
MODEL_MAPPINGS = {
    "gpt-4o": "gemini-2.5-pro-exp-03-25",
    "gpt-4o-2024-08-06": "gemini-2.5-pro-exp-03-25",
    "default": "gemini-2.5-pro-exp-03-25",
    "gpt-3.5-turbo": "gemini-2.5-pro-exp-03-25"
}

# API request settings
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "120"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File logging settings
LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "1") == "1"
LOG_DIR = os.environ.get("LOG_DIR", "logs")
if LOG_TO_FILE and not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.info(f"Created log directory: {LOG_DIR}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_to_file(content, prefix="response", include_timestamp=True):
    """Log content to a text file for debugging"""
    if not LOG_TO_FILE:
        return
    
    try:
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{LOG_DIR}/{prefix}_{timestamp}.txt"
        
        # Format the content
        if isinstance(content, dict) or isinstance(content, list):
            formatted_content = json.dumps(content, indent=2)
        else:
            formatted_content = str(content)
            
        # Add a timestamp at the top of the file
        if include_timestamp:
            header = f"=== {prefix.upper()} LOG - {datetime.datetime.now().isoformat()} ===\n\n"
        else:
            header = ""
            
        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header + formatted_content)
            
        logger.info(f"Logged {prefix} content to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error logging to file: {str(e)}")
        return None

def map_openai_model_to_gemini(model_name):
    """Map OpenAI model name to Gemini model name"""
    return MODEL_MAPPINGS.get(model_name, MODEL_MAPPINGS["default"])

def convert_openai_messages_to_gemini(messages):
    """Convert OpenAI message format to Gemini format"""
    formatted_messages = []
    
    # Process system message first if present
    system_content = None
    for message in messages:
        if message.get('role') == 'system':
            system_content = message.get('content', '')
            break
    
    # Process other messages
    for message in messages:
        role = message.get('role')
        content = message.get('content', '')
        
        if role == 'system':
            # System messages handled separately
            continue
        elif role == 'user':
            formatted_messages.append({'role': 'user', 'parts': [{'text': content}]})
        elif role == 'assistant':
            formatted_messages.append({'role': 'model', 'parts': [{'text': content}]})
        elif role == 'function':
            # Pass through function responses as user messages with clear labeling
            formatted_messages.append({
                'role': 'user', 
                'parts': [{'text': f"Function result: {message.get('name', '')}\n{content}"}]
            })
    
    return formatted_messages, system_content

def gemini_streaming_chunk_to_openai_chunk(chunk_text, model_name):
    """Convert a Gemini streaming chunk to OpenAI streaming format"""
    # Create a delta with the content
    delta = {"content": chunk_text}
    
    chunk = {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None
        }]
    }
    
    return chunk

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
            return None
            
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

def load_system_prompt():
    """Load the system prompt from CursorSystemPrompt.md file"""
    try:
        # Try to find the file in the current directory or one level up
        file_paths = [
            "CursorSystemPrompt.md",
            "../CursorSystemPrompt.md",
            "CursorCustomModels/CursorSystemPrompt.md",
            "../CursorCustomModels/CursorSystemPrompt.md"
        ]
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    logger.info(f"Loaded system prompt from {file_path}")
                    return file.read()
        
        # If no file found, return default instructions
        logger.warning("CursorSystemPrompt.md not found. Using default instructions.")
        return """
# Default System Instructions
You are a powerful AI coding assistant. 
Follow user's instructions carefully.
Use tools when appropriate.
        """
    except Exception as e:
        logger.error(f"Error loading system prompt: {str(e)}")
        return """
# Default System Instructions (Error Recovery)
You are a powerful AI coding assistant.
Follow user's instructions carefully.
Use tools when appropriate.
        """

# Load the Cursor system prompt at startup
CURSOR_SYSTEM_PROMPT = load_system_prompt()
logger.info(f"Loaded Cursor system prompt ({len(CURSOR_SYSTEM_PROMPT)} characters)")

# ============================================================================
# ROUTE HANDLERS
# ============================================================================

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Expose-Headers', 'X-Request-ID, openai-organization, openai-processing-ms, openai-version')
    return response

# Handle OPTIONS requests for all routes
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle OPTIONS requests for all routes"""
    response = make_response('')
    response.status_code = 200
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin')
    response.headers.add('Access-Control-Expose-Headers', 'X-Request-ID, openai-organization, openai-processing-ms, openai-version')
    return response

def process_chat_request():
    """Common handler for chat completion requests"""
    try:
        # Get request data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        # Log the request data
        log_to_file(data, "request")
        
        # Extract request parameters
        messages = data.get('messages', [])
        model_name = data.get('model', 'gpt-4o')  # Store original model name
        stream = data.get('stream', False)
        
        logger.info(f"Processing chat request with model: {model_name}, stream: {stream}")
        
        # Check if we need to add the system prompt
        has_system_message = False
        for message in messages:
            if message.get('role') == 'system':
                # Append our Cursor system prompt to the existing system message
                existing_content = message.get('content', '')
                message['content'] = existing_content + "\n\n" + CURSOR_SYSTEM_PROMPT
                has_system_message = True
                logger.info("Added Cursor system prompt to existing system message")
                break
        
        # If no system message exists, add one with our Cursor system prompt
        if not has_system_message:
            messages.insert(0, {
                "role": "system",
                "content": CURSOR_SYSTEM_PROMPT
            })
            logger.info("Added new system message with Cursor system prompt")
            
        # Convert messages to Gemini format
        gemini_messages, system_content = convert_openai_messages_to_gemini(messages)
        
        # Get the Gemini model name
        gemini_model_name = map_openai_model_to_gemini(model_name)
        
        # Configure the model
        try:
            gemini_model = genai.GenerativeModel(gemini_model_name)
            logger.info(f"Created Gemini model: {gemini_model_name}")
        except Exception as e:
            logger.error(f"Error creating Gemini model: {str(e)}")
            # Fallback to default model
            gemini_model_name = MODEL_MAPPINGS["default"]
            gemini_model = genai.GenerativeModel(gemini_model_name)
            logger.info(f"Falling back to default model: {gemini_model_name}")
        
        # Add system message if present
        if system_content:
            gemini_model.system_instruction = system_content
        
        # Create the chat
        chat = gemini_model.start_chat(history=gemini_messages)
        
        if stream:
            # Handle streaming
            def generate_streaming_response():
                try:
                    # Use send_message with a neutral prompt
                    response_stream = chat.send_message(
                        "Continue the conversation.",
                        stream=True
                    )
                    
                    # For logging purposes
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    stream_log_filename = f"{LOG_DIR}/stream_{timestamp}.txt"
                    all_chunks = []
                    
                    for chunk in response_stream:
                        chunk_text = chunk.text if hasattr(chunk, 'text') else ""
                        
                        # Log the raw chunk
                        if LOG_TO_FILE:
                            all_chunks.append(f"RAW CHUNK: {chunk_text}")
                        
                        if chunk_text:
                            # Convert chunk to OpenAI format
                            openai_chunk = gemini_streaming_chunk_to_openai_chunk(
                                chunk_text, 
                                model_name  # Use original model name
                            )
                            
                            # Log the formatted chunk
                            if LOG_TO_FILE:
                                all_chunks.append(f"FORMATTED CHUNK: {json.dumps(openai_chunk)}")
                            
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                    
                    # Send a final chunk with finish_reason
                    final_chunk = {
                        "id": f"chatcmpl-{str(uuid.uuid4())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,  # Use original model name
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    
                    # Log the final chunk
                    if LOG_TO_FILE:
                        all_chunks.append(f"FINAL CHUNK: {json.dumps(final_chunk)}")
                    
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    
                    # Send [DONE] marker
                    if LOG_TO_FILE:
                        all_chunks.append("DONE MARKER: data: [DONE]")
                    
                    yield "data: [DONE]\n\n"
                    
                    # Write all chunks to a single file
                    if LOG_TO_FILE:
                        try:
                            with open(stream_log_filename, "w", encoding="utf-8") as f:
                                header = f"=== STREAMING RESPONSE LOG - {datetime.datetime.now().isoformat()} ===\n\n"
                                f.write(header)
                                f.write("\n\n".join(all_chunks))
                            logger.info(f"Logged streaming response to {stream_log_filename}")
                        except Exception as e:
                            logger.error(f"Error logging streaming response: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error in Gemini streaming: {str(e)}")
                    
                    error_chunk = {
                        "id": f"chatcmpl-{str(uuid.uuid4())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,  # Use original model name
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n\nError: {str(e)}"},
                            "finish_reason": "stop"
                        }]
                    }
                    
                    # Log the error
                    log_to_file({
                        "error": str(e),
                        "error_chunk": error_chunk
                    }, "stream_error")
                    
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return Response(
                generate_streaming_response(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Content-Type': 'text/event-stream'
                }
            )
        
        else:
            # Handle non-streaming
            try:
                response = chat.send_message("Continue the conversation.")
                
                # Get content from response
                content = response.text
                
                # Log the raw response from Gemini
                log_to_file({
                    "raw_content": content,
                    "response_object": str(response)
                }, "gemini_raw_response")
                
                # Format response in OpenAI format
                openai_response = {
                    "id": f"chatcmpl-{str(uuid.uuid4())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,  # Use original model name
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                
                # Log the formatted OpenAI response
                log_to_file(openai_response, "openai_formatted_response")
                
                return jsonify(openai_response)
            except Exception as e:
                logger.error(f"Error in non-streaming response: {str(e)}")
                # Log the error
                log_to_file({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }, "non_streaming_error")
                raise  # Re-raise to be caught by the outer try/except
            
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        
        # Return an error response in OpenAI format
        error_response = {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name if 'model_name' in locals() else "unknown",  # Use model_name instead of model object
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}\n\nPlease try again or check your API key configuration."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # Log the error response
        log_to_file({
            "error": str(e),
            "error_response": error_response
        }, "error_response")
        
        if 'stream' in locals() and stream:
            def generate_error_stream():
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(
                generate_error_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Content-Type': 'text/event-stream'
                }
            )
        else:
            return jsonify(error_response)

@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completions():
    """Handle OpenAI-compatible chat completion endpoint"""
    return process_chat_request()

@app.route('/chat/completions', methods=['POST'])
def cursor_chat_completions():
    """Handle Cursor-specific chat completion endpoint"""
    return process_chat_request()

@app.route('/<path:path>/chat/completions', methods=['POST'])
def any_chat_completions(path):
    """Handle chat completion with custom path prefix"""
    return process_chat_request()

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Return a list of available models"""
    models = [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        },
        {
            "id": "gpt-4o-2024-08-06",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        }
    ]
    
    return jsonify({
        "object": "list",
        "data": models
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/', methods=['GET'])
def home():
    """Render home page with information about the proxy"""
    return """
    <html>
    <head>
        <title>Simple Gemini Proxy for Cursor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        </style>
    </head>
    <body>
        <h1>Simple Gemini Proxy for Cursor</h1>
        <p>This server acts as a simple proxy between Cursor and Google's Gemini API.</p>
        
        <h2>Available Endpoints</h2>
        <ul>
            <li>/v1/chat/completions - Standard OpenAI-compatible chat completion endpoint</li>
            <li>/chat/completions - Cursor-specific chat completion endpoint</li>
            <li>/v1/models - List available models</li>
            <li>/health - Health check endpoint</li>
        </ul>
    </body>
    </html>
    """

# ============================================================================
# MAIN FUNCTION
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    use_ngrok = os.environ.get("USE_NGROK", "0") == "1"
    
    print(f"Starting simple Gemini proxy server on {host}:{port}")
    
    # Start ngrok if configured to do so
    if use_ngrok:
        logger.info("Starting ngrok tunnel...")
        public_url = start_ngrok(port)
        if not public_url:
            logger.warning("Failed to start ngrok. Continuing with local server only.")
    
    # Start the server
    serve(app, host=host, port=port) 