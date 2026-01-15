# BioBot Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [API Integration](#api-integration)
6. [Architecture](#architecture)
7. [Troubleshooting](#troubleshooting)
8. [Development](#development)

## Overview

BioBot is a voice-controlled AI assistant with OpenWebUI integration. Capture screenshots and ask questions using voice commands.

**Target Users:** Datacenter technicians and field workers who need hands-free access to technical documentation.

**Key Features:**
- Push-to-talk voice recording
- Screenshot capture (focused window or full screen)
- Speech-to-text (OpenAI Whisper, Groq, or Local Whisper)
- OpenWebUI integration with RAG support
- Vision model support for image analysis
- Conversation history management

## Installation

### Prerequisites

- macOS (Monterey or later)
- Python 3.11+
- Open WebUI instance
- Microphone and screen recording permissions

### Quick Start

```bash
# Clone repository
git clone https://github.com/Braryai/biobot.git
cd biobot/biobot-client

# Install dependencies
pip install -r requirements.txt

# Configure
cp config.py.example config.py
nano config.py  # Edit with your credentials

# Run
python biobot_voice.py
```

## Configuration

### Required Settings

Edit `config.py`:

```python
# OpenWebUI Server
OPENWEBUI_URL = "http://your-server-url"
OPENWEBUI_TOKEN = "your-token"

# Speech-to-Text
OPENAI_API_KEY = "your-key"  # or use Groq/Local Whisper
```

### Optional Settings

```python
# Knowledge Base (RAG)
KNOWLEDGE_ID = "your-kb-id"

# Model Selection
DEFAULT_MODEL = "llama3.1:8b"

# Alternative STT Providers
USE_GROQ_STT = False
GROQ_API_KEY = "your-groq-key"

USE_LOCAL_WHISPER = True  # Offline, free
LOCAL_WHISPER_MODEL = "base"

# Text-to-Speech (Optional)
USE_TTS = False
TOTALGPT_API_KEY = "your-key"
TTS_VOICE = "af_bella"

# Keyboard Shortcuts
TRIGGER_KEY_WITH_SCREENSHOT = "cmd_r"
TRIGGER_KEY_AUDIO_ONLY = "shift_r"
```

### Getting Credentials

**OpenWebUI Token:**
1. Open WebUI → Settings → Account → API Keys
2. Create new key → Copy

**Knowledge Base ID:**
1. Workspace → Knowledge
2. Select your knowledge base
3. Copy ID from URL or settings

**STT API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Groq: https://console.groq.com/keys

## Usage

### Operating Modes

**Voice + Screenshot Mode** (Default: Right Cmd)
1. Press and hold trigger key
2. Speak your question
3. Release to process
4. Screenshot captured automatically

**Voice Only Mode** (Default: Right Shift)
1. Press and hold trigger key
2. Speak your question
3. Release to process

### Keyboard Shortcuts

- `cmd_r`: Voice + Screenshot
- `shift_r`: Voice only
- `esc` during recording: Reset conversation

### Example Queries

**With Screenshot:**
- "What is this component?"
- "How do I configure this switch?"
- "What cables should I use?"

**Without Screenshot:**
- "Server maintenance procedure"
- "VLAN cable color codes"
- "Power requirements for rack 12"

## API Integration

### OpenWebUI Endpoints

BioBot uses the following REST API endpoints:

**Chat Completions:**
```
POST /api/chat/completions
Content-Type: application/json
Authorization: Bearer {token}

{
  "model": "llama3.1:8b",
  "messages": [
    {
      "role": "user",
      "content": "your question",
      "images": ["base64..."]  // Optional
    }
  ],
  "knowledge_id": "your-kb-id"
}
```

**File Upload:**
```
POST /api/v1/files/
Content-Type: multipart/form-data

file: <binary screenshot data>
```

**Chat Management:**
```
GET /api/v1/chats/{id}      # Retrieve history
POST /api/v1/chats/{id}     # Save messages with attachments
GET /api/models             # List available models
```

### Image Upload Workflow

1. **Upload Screenshot:** `POST /api/v1/files/` (multipart)
   - Returns `file_id`

2. **Query Model:** `POST /api/chat/completions` (inline base64)
   - Get response from model

3. **Save to Chat:** `POST /api/v1/chats/{id}`
   - Include attachment metadata:
   ```json
   {
     "type": "image",
     "file_id": "...",
     "url": "/api/v1/files/...",
     "thumbnail_url": "/api/v1/files/.../thumbnail",
     "filename": "screenshot.png",
     "content_type": "image/png"
   }
   ```

### Important Notes

- Chat ID is NOT returned in HTTP streaming/non-streaming responses
- Must create chat explicitly on first message
- Frontend requires manual save for messages to appear
- Auto-refresh not implemented (requires F5)

## Architecture

### System Flow

```
User Voice Input
    ↓
Audio Recording (sounddevice)
    ↓
Speech-to-Text (Whisper/Groq)
    ↓
Screenshot Capture (screencapture)
    ↓
OpenWebUI API
    ↓
LLM + RAG (Knowledge Base)
    ↓
Response Display
    ↓
Optional TTS
```

### Technology Stack

**Client:**
- Python 3.11+
- pynput (keyboard input)
- sounddevice (audio recording)
- OpenAI SDK (Whisper STT)
- httpx (HTTP client)
- PyObjC (macOS integration)

**Server:**
- Open WebUI
- Ollama (LLM runtime)
- ChromaDB (vector storage)

**External Services:**
- OpenAI Whisper API (STT)
- Groq API (alternative STT)
- TotalGPT API (optional TTS)

## Troubleshooting

### Connection Issues

**"Could not connect to Open WebUI"**
- Verify server is running
- Check `OPENWEBUI_URL` in config.py
- Validate `OPENWEBUI_TOKEN`
- Test: `python test_openwebui.py`

### Model Issues

**"Model not found"**
- List models: `python list_models.py`
- Verify `DEFAULT_MODEL` name
- Check TotalGPT/Ollama endpoint in OpenWebUI

### Audio Issues

**"No audio recorded"**
- Grant microphone permissions to Terminal
- Check System Preferences → Audio Input
- Test different `AUDIO_SAMPLE_RATE` values

### Screenshot Issues

**"Screenshot capture failed"**
- Grant screen recording permissions to Terminal
- Verify focused window exists
- Falls back to full screen automatically

### Frontend Issues

**Messages not appearing in OpenWebUI**
- Manual F5 refresh required
- Verify chat is saved via `/api/v1/chats/{id}`
- Check browser console for errors

## Development

### Project Structure

```
biobot/
├── biobot-client/          # Main voice client
│   ├── biobot_voice.py     # Core implementation
│   ├── config.py.example   # Configuration template
│   ├── requirements.txt    # Dependencies
│   ├── test_openwebui.py   # Connection tests
│   └── list_models.py      # Model listing utility
├── docs/                   # Documentation
├── .github/workflows/      # CI/CD
└── CHANGELOG.md           # Version history
```

### Testing

```bash
cd biobot-client

# Test OpenWebUI connection
python test_openwebui.py

# List available models
python list_models.py

# Debug model issues
python debug_models.py

# Test chat creation
python test_chat_creation.py
```

### CI/CD

GitHub Actions workflow runs on push:
- Python syntax validation
- flake8 linting
- Code compilation checks

### Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Update CHANGELOG.md
5. Submit pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public functions
- No emojis in code
- English comments only
- Meaningful variable names

## License

MIT License - see [LICENSE](../LICENSE) file.

## Support

For issues and contributions: https://github.com/Braryai/biobot
