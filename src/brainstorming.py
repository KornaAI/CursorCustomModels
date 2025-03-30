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
import subprocess  # Added for ngrok
import datetime  # For timestamping logs
import threading  # Added for CLI input
import queue      # Added for message passing

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# OpenAI API endpoints that we'll intercept
OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"
CURSOR_CHAT_ENDPOINT = "/chat/completions"

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

# Global queue for passing messages from CLI to stream handler
message_queue = queue.Queue()

# ============================================================================
# HELPER FUNCTIONS & CLI THREAD
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

def create_openai_chunk(chunk_text, model_name):
    """Creates an OpenAI-compatible streaming chunk."""
    delta = {"content": chunk_text}
    chunk = {
        "id": f"chatcmpl-brainstorm-{str(uuid.uuid4())}", # Custom ID prefix
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name, # Use the model requested by the client
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
            "role" = "user"
            "content" = "Start brainstorming..."
        }}
    )
    "model" = "gpt-4o",
    "stream" = $true
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
      {{"role": "user", "content": "Start brainstorming..."}}
    ],
    "stream": true
  }}'
                """)
                
                # Print instructions for Cursor
                print("\nTo configure Cursor:")
                print(f"1. Set the OpenAI API base URL to: {public_url}")
                print("2. Use 'gpt-4o' or any other model name Cursor sends.")
                print("3. Set any API key (it won't be checked)")
                print("4. Ensure 'stream' is enabled in the request (usually default in Cursor)")
                print("5. Check the ngrok inspector at http://localhost:4040 to debug traffic")
                
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

def cli_input_loop(q):
    """Thread function to read CLI input and put it in the queue."""
    print("\nCLI Input enabled. Type messages here and press Enter to send them to the stream.")
    print("Server logs will appear above. Type 'exit' or 'quit' in CLI to stop server (gracefully).")
    while True:
        try:
            message = input("> ")
            if message.lower() in ['exit', 'quit']:
                print("CLI requesting server shutdown...")
                q.put("SERVER_SHUTDOWN_REQUEST") # Signal for graceful shutdown
                 # Give time for signal to be processed if needed
                time.sleep(0.5)
                # A more robust solution would involve signaling Waitress to stop.
                # For simplicity, we'll exit the process here. Waitress might need Ctrl+C.
                os._exit(0) # Force exit if Waitress doesn't handle the signal
                break
            q.put(message)
        except EOFError: # Handle Ctrl+D or pipe closing
            logger.info("CLI input EOF received, stopping input thread.")
            break
        except Exception as e:
            logger.error(f"Error in CLI input thread: {e}")
            time.sleep(1) # Avoid busy-looping on error

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
    """Handles chat completion requests by streaming dots and CLI messages."""
    request_id = f"breq-{str(uuid.uuid4())}" # Unique ID for brainstorming request
    logger.info(f"[{request_id}] Received chat request")
    
    try:
        # Get request data
        if not request.is_json:
            logger.warning(f"[{request_id}] Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        # Log the request data
        log_filename = log_to_file(data, f"request_{request_id}")
        logger.info(f"[{request_id}] Logged request data to {log_filename}")
        
        # Extract request parameters
        model_name = data.get('model', 'gpt-4o')  # Get model name from request
        stream = data.get('stream', False)
        messages = data.get('messages', []) # Log messages for context if needed
        
        # Log basic info
        logger.info(f"[{request_id}] Processing request: model={model_name}, stream={stream}, messages_count={len(messages)}")
        if messages:
             logger.info(f"[{request_id}] Last user message: {messages[-1].get('content', 'N/A')[:100]}...")

        if stream:
            # Handle streaming: send dots, newlines, and CLI messages
            def generate_dot_stream():
                logger.info(f"[{request_id}] Starting dot/message stream for model {model_name}")
                stream_start_time = time.time()
                dot_count = 0
                dots_since_newline = 0
                global message_queue # Access the global queue

                try:
                    while True:
                        # 1. Check for CLI messages (non-blocking)
                        try:
                            cli_message = message_queue.get_nowait()
                            logger.info(f"[{request_id}] Got CLI message: '{cli_message}'")

                            if cli_message == "SERVER_SHUTDOWN_REQUEST":
                                logger.warning(f"[{request_id}] Shutdown requested via CLI message queue. Stopping stream.")
                                yield f"data: {json.dumps(create_openai_chunk('[Server shutdown requested via CLI]', model_name))}\n\n"
                                # Send final chunk before breaking
                                final_chunk = {
                                    "id": f"chatcmpl-{str(uuid.uuid4())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                                }
                                yield f"data: {json.dumps(final_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                break # Exit the while loop

                            # Format and send the CLI message (Removed formatting)
                            # formatted_cli_msg = f"\n\n>>> {cli_message}\n\n" # Old formatting
                            # Send the raw message, maybe add a newline after for separation
                            formatted_cli_msg = f"{cli_message}\n" 
                            chunk = create_openai_chunk(formatted_cli_msg, model_name)
                            yield f"data: {json.dumps(chunk)}\n\n"
                            dots_since_newline = 0 # Reset newline counter after message

                        except queue.Empty:
                            # No message from CLI, continue with dots/newlines
                            pass
                        except Exception as q_err:
                             logger.error(f"[{request_id}] Error processing queue message: {q_err}")


                        # 2. Send dot or newline
                        content_to_send = ""
                        if dots_since_newline >= 25:
                            content_to_send = "\n"
                            dots_since_newline = 0
                        else:
                            content_to_send = "."
                            dots_since_newline += 1
                            dot_count += 1

                        chunk = create_openai_chunk(content_to_send, model_name)
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
                        # Log progress occasionally
                        if dot_count % 100 == 0 and dot_count > 0:
                             logger.debug(f"[{request_id}] Sent {dot_count} dots...")
                             
                        # Small delay to prevent overwhelming the client/network
                        # Also yields control briefly, helping responsiveness
                        time.sleep(0.05) 
                        
                except GeneratorExit:
                    # This happens when the client disconnects
                    logger.info(f"[{request_id}] Client disconnected stream after {time.time() - stream_start_time:.2f}s and {dot_count} dots.")
                except Exception as e:
                    logger.error(f"[{request_id}] Error during streaming: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Attempt to send an error chunk if possible
                    try:
                        error_delta = {"content": f"\n\n[STREAM ERROR: {str(e)}]"}
                        error_chunk = {
                            "id": f"chatcmpl-error-{str(uuid.uuid4())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": error_delta, "finish_reason": "stop"}]
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as nested_e:
                         logger.error(f"[{request_id}] Failed to send error chunk: {str(nested_e)}")
                finally:
                    logger.info(f"[{request_id}] Dot/message stream finished.")

            return Response(
                generate_dot_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no', # Important for streaming
                    'Content-Type': 'text/event-stream'
                }
            )
        
        else:
            # Handle non-streaming: return a fixed response
            logger.info(f"[{request_id}] Handling non-streaming request for model {model_name}")
            openai_response = {
                "id": f"chatcmpl-brainstorm-ns-{str(uuid.uuid4())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a non-streaming response from the brainstorming server."
                    },
                    "finish_reason": "stop"
                }],
                "usage": { # Dummy usage data
                    "prompt_tokens": 0,
                    "completion_tokens": 10,
                    "total_tokens": 10
                }
            }
            log_to_file(openai_response, f"response_{request_id}_nonstream")
            return jsonify(openai_response)
            
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled error processing chat request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log the error
        log_to_file({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "request_data": data if 'data' in locals() else "Not available"
        }, f"error_{request_id}")

        # Return an error response in OpenAI format
        error_response = {
            "id": f"chatcmpl-error-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name if 'model_name' in locals() else "unknown",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}\n\nFailed to process brainstorming request."
                },
                "finish_reason": "error" # Use 'error' finish reason
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # Attempt to return error in requested format (stream vs non-stream)
        if 'stream' in locals() and stream:
            def generate_error_stream():
                try:
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as stream_err:
                     logger.error(f"[{request_id}] Error generating error stream: {stream_err}")

            return Response(
                generate_error_stream(),
                status=500, # Internal Server Error
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Content-Type': 'text/event-stream'
                }
            )
        else:
            return jsonify(error_response), 500

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
    logger.info(f"Received request on custom path: /{path}/chat/completions")
    return process_chat_request()

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Return a list of available models (mimics OpenAI)"""
    logger.info("Received request for /v1/models")
    # Crucially, list gpt-4o so Cursor can select it
    models = [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai" # Pretend ownership
        },
        {
            "id": "gpt-4o-2024-08-06",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        },
         # Add others if Cursor might try them, though this server ignores the choice
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
    """Render home page with information about the brainstorming proxy"""
    return """
    <html>
    <head>
        <title>Brainstorming Dot Streamer for Cursor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            code { background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Brainstorming Dot Streamer for Cursor</h1>
        <p>This server mimics the OpenAI API but responds to chat completion requests by streaming an endless sequence of dots (<code>.</code>).</p>
        <p>It's designed for testing streaming integrations or simulating a continuous "thinking" process in Cursor.</p>
        
        <h2>Configuration</h2>
        <p>To use this with Cursor:</p>
        <ol>
            <li>Run this server (optionally with <code>USE_NGROK=1</code> environment variable for a public URL).</li>
            <li>In Cursor's settings, set the "OpenAI API Base URL" to the address of this server (e.g., <code>http://localhost:5000</code> or the ngrok URL).</li>
            <li>Set the model to <code>gpt-4o</code> or any other model listed by the <code>/v1/models</code> endpoint.</li>
            <li>Use any value for the API Key.</li>
        </ol>
        
        <h2>Available Endpoints</h2>
        <ul>
            <li><code>/v1/chat/completions</code> (POST) - Accepts OpenAI chat requests, streams dots if <code>stream: true</code>.</li>
            <li><code>/chat/completions</code> (POST) - Cursor-specific endpoint, same behavior.</li>
             <li><code>/&lt;path&gt;/chat/completions</code> (POST) - Handles custom path prefixes, same behavior.</li>
            <li><code>/v1/models</code> (GET) - Lists available models (e.g., <code>gpt-4o</code>).</li>
            <li><code>/health</code> (GET) - Health check.</li>
        </ul>
         <p>Check the server logs and optionally the ngrok inspector (<code>http://localhost:4040</code> if using ngrok) for request details.</p>
    </body>
    </html>
    """

# ============================================================================
# MAIN FUNCTION
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5001")) # Use a different default port than gemini.py
    host = os.environ.get("HOST", "0.0.0.0")
    use_ngrok = os.environ.get("USE_NGROK", "0") == "1"
    
    print(f"Starting Brainstorming Dot Streamer server on {host}:{port}")
    print("This server mimics OpenAI API, streams dots with newlines, and accepts CLI input.")
    
    # Start ngrok if configured to do so
    if use_ngrok:
        logger.info("Starting ngrok tunnel...")
        public_url = start_ngrok(port)
        if not public_url:
            logger.warning("Failed to start ngrok. Continuing with local server only.")
    else:
         print(f"Configure Cursor Base URL to: http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}")

    # Start the CLI input thread
    cli_thread = threading.Thread(target=cli_input_loop, args=(message_queue,), daemon=True)
    cli_thread.start()
    logger.info("Started CLI input thread.")

    # Start the server using waitress
    try:
        serve(app, host=host, port=port)
    except KeyboardInterrupt:
         logger.info("KeyboardInterrupt received, stopping server.")
    finally:
         logger.info("Server shutting down.")
         # Optionally signal the CLI thread if it wasn't a daemon or handle cleanup 