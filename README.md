# BioBot

Voice-controlled AI assistant with OpenWebUI integration. Capture screenshots and ask questions using voice commands.

## Overview

BioBot enables hands-free interaction with AI models through voice commands and screenshot capture. Built for datacenter technicians and field workers who need instant access to technical information while working with equipment.

**Key Features:**
- Push-to-talk voice recording
- Automatic screenshot capture  
- Local Speech-to-Text (offline Whisper)
- OpenWebUI integration with RAG support
- Vision model support for image analysis
- Configurable system prompts via voice
- Real-time dashboard with debug logs
- Conversation history management
- Configurable keyboard shortcuts

## Installation

### Prerequisites

- macOS (Monterey or later)
- Python 3.11+
- Open WebUI instance
- Microphone and screen recording permissions

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Braryai/biobot.git
cd biobot/biobot-client
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the client:
```bash
cp config.py.example config.py
nano config.py
```

Edit `config.py` with your credentials:
- `OPENWEBUI_URL`: Your server URL
- `OPENWEBUI_TOKEN`: Your API token
- `OPENAI_API_KEY`: Your STT API key

See `config.py.example` for all available options.

4. Run BioBot:
```bash
python biobot_voice.py
```


## Usage

### Basic Operation

Two modes activated by keyboard shortcuts:

**Voice + Screenshot Mode** (Default: Right Cmd)
1. Press and hold trigger key
2. Speak your question
3. Release to process
4. Screenshot captured automatically

**Voice Only Mode** (Default: Right Shift)
1. Press and hold trigger key
2. Speak your question
3. Release to process

### Voice Commands

Available voice commands:
- "set system prompt [text]" - Configure AI behavior
- "new chat" - Start a new conversation
- "delete last" - Remove last message

### Keyboard Shortcuts

Configurable in `config.py`:
- `cmd_r`: Voice + Screenshot
- `shift_r`: Voice only
- `esc`: Reset conversation

## Configuration

All settings are in `config.py`. Copy from `config.py.example` and update:

**Required:**
```python
OPENWEBUI_URL = "http://your-server-url"
OPENWEBUI_TOKEN = "your-token-here"
OPENAI_API_KEY = "your-key-here"
```

**Optional:**
```python
KNOWLEDGE_ID = "your-kb-id"
DEFAULT_MODEL = "Doctor-Shotgun-L3.3-70B-Magnum-v4-SE"  # Or any model from OpenWebUI
USE_LOCAL_WHISPER = True  # Recommended: free offline STT
LOCAL_WHISPER_MODEL = "base.en"  # or "small", "medium", "large-v3"
USE_GROQ_STT = False  # Alternative: Groq API
USE_TTS = False  # Optional text-to-speech
```

See `config.py.example` for complete documentation.

**Recommended Model:**
For best system prompt support, use `Doctor-Shotgun-L3.3-70B-Magnum-v4-SE` or similar high-quality models.


## Architecture

```
Voice Input → STT → OpenWebUI API → LLM Response
Screenshot → Upload → Vision Model Analysis
```

## API Integration

Uses OpenWebUI REST API:
- `POST /api/chat/completions` - Send queries
- `POST /api/v1/files/` - Upload screenshots
- `GET /api/v1/chats/{id}` - Retrieve history
- `POST /api/v1/chats/{id}` - Save messages
- `GET /api/models` - List models

## Dashboard

BioBot includes a real-time dashboard with:
- Connection status and statistics
- Live activity log (last transcript and response)
- Debug logs panel for troubleshooting
- System prompt status
- Processing progress bar
- Keyboard shortcuts reference

The dashboard updates automatically and provides visual feedback for all operations.

## Troubleshooting

**Connection Issues:**
- Verify OpenWebUI is running
- Check `OPENWEBUI_URL` and `OPENWEBUI_TOKEN` in config.py
- Check dashboard for connection status

**Audio Issues:**
- Grant microphone permissions to Terminal
- Check System Preferences audio input device
- Verify audio levels in dashboard during recording

**Screenshot Issues:**
- Grant screen recording permissions to Terminal
- Grant accessibility permissions to Terminal
- Verify window focus exists

**System Prompt Not Working:**
- Check debug logs panel in dashboard
- Verify system prompt was set (shows in dashboard)
- Ensure model supports system prompts (use recommended model)

## Development

### Project Structure

```
biobot/
├── biobot-client/          # Voice client
│   ├── biobot_voice.py     # Main implementation
│   ├── config.py.example   # Configuration template
│   └── requirements.txt    # Dependencies
├── biobot-api/            # API services (future)
└── docs/                  # Documentation
```

### Utilities

```bash
cd biobot-client
python list_models.py     # List available models in OpenWebUI
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Support

Visit the [GitHub repository](https://github.com/Braryai/biobot) for issues and contributions.


