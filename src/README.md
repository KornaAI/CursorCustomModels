# Gemini Proxy for Cursor

This proxy allows you to use Google's Gemini models with Cursor IDE, providing full compatibility with Cursor's tools and features.

## Features

- 🔄 Full OpenAI API compatibility for Cursor
- 🚀 Direct integration with Google's Gemini API
- 💬 Proper streaming support for smooth responses
- 🛠️ Tool calling support for Cursor's file operations
- 📝 Automatic model mapping from OpenAI to Gemini models

## Setup Instructions

1. **Install Dependencies**

   ```bash
   pip install -r ../requirements.txt
   ```

2. **Configure API Key**

   Edit the `.env` file and add your Google API key:

   ```
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

   You can get a Google AI API key from [Google AI Studio](https://makersuite.google.com/).

3. **Run the Proxy**

   ```bash
   python gemini_proxy.py
   ```

   The server will start on port 5000 by default.

## Configure Cursor

1. First, you must verify Cursor with a real OpenAI API key (this is required to unlock Custom API Mode).
2. After verification, change the "Base URL" to:
   - `http://localhost:5000` (for local use)
   - Or your ngrok URL (for remote access)
3. Click "Verify" again to test your proxy.
4. Start using Cursor with Gemini models!

## Remote Access with ngrok

You can expose your local proxy to the internet using ngrok:

1. Install ngrok from [https://ngrok.com/download](https://ngrok.com/download)
2. Edit the `.env` file and set `USE_NGROK=1`
3. Run the proxy as normal
4. The ngrok public URL will be displayed in the console
5. Use this URL as the "Base URL" in Cursor settings

## Available Endpoints

- `/v1/chat/completions` - Standard OpenAI-compatible endpoint
- `/chat/completions` - Cursor-specific endpoint
- `/agent` - Endpoint with agent mode always enabled

## Model Mapping

The proxy automatically maps OpenAI model names to Gemini models:

- `gpt-4o` → `gemini-1.5-pro-latest`
- `gpt-3.5-turbo` → `gemini-1.5-flash-latest`

You can customize these mappings in the code or your `.env` file.

## Troubleshooting

- **API Key Issues**: Make sure your Google API key is correctly set in the `.env` file
- **Model Errors**: Verify that the Gemini models are available in your region
- **Connection Issues**: Check that the proxy is running at the URL configured in Cursor

## License

This project is provided as-is without warranty. Use at your own risk. 