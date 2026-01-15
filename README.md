# BioBot

A voice-controlled AI assistant for datacenter technicians enabling hands-free access to technical documentation through Open WebUI with RAG (Retrieval-Augmented Generation).

## Overview

BioBot provides datacenter technicians with instant, hands-free access to technical documentation while working with equipment. Using push-to-talk voice commands and optional screenshot capture, technicians can query knowledge bases and receive accurate responses with source citations.

**Key Features:**
- Push-to-talk voice recording with visual feedback
- Smart window or full-screen screenshot capture
- Speech-to-text transcription (OpenAI Whisper or Groq)
- Direct Open WebUI API integration with RAG support
- Vision model support for screenshot analysis
- Conversation history management
- Optional text-to-speech responses
- Configurable keyboard shortcuts

## Installation

### Prerequisites

- macOS (tested on Monterey and later)
- Python 3.11 or higher
- Open WebUI instance with accessible API
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

Required configuration:
- `OPENWEBUI_URL`: Your Open WebUI server URL
- `OPENWEBUI_TOKEN`: API token from Open WebUI
- `OPENAI_API_KEY`: For speech-to-text transcription

Optional configuration:
- `KNOWLEDGE_ID`: Knowledge base ID for RAG queries
- `DEFAULT_MODEL`: Model to use for queries
- `USE_GROQ_STT`: Enable Groq for faster/cheaper transcription
- `USE_TTS`: Enable text-to-speech responses

4. Run BioBot:
```bash
python biobot_voice.py
```


## Usage

### Basic Operation

BioBot operates in two modes activated by keyboard shortcuts:

**Mode 1: Voice + Screenshot** (Default: Right Cmd)
1. Press and hold the trigger key
2. Speak your question
3. Release the key to process
4. Screenshot is automatically captured
5. Response displays with citations

**Mode 2: Voice Only** (Default: Right Shift)
1. Press and hold the trigger key
2. Speak your question
3. Release the key to process
4. No screenshot captured

### Example Queries

**With Screenshot:**
- "What is this component?"
- "How do I configure this switch?"
- "What cables should I use for this setup?"

**Without Screenshot:**
- "What is the procedure for server maintenance?"
- "Show me the cable color codes for VLANs"
- "What are the power requirements for rack 12?"

### Keyboard Shortcuts

Default shortcuts (configurable in `config.py`):
- `cmd_r`: Voice + Screenshot mode
- `shift_r`: Voice only mode
- `cmd+r` (during recording): Reset conversation

## Configuration


### Open WebUI Setup

1. Access your Open WebUI instance
2. Navigate to Settings → Account → API Keys
3. Generate a new API key
4. Copy the key to `OPENWEBUI_TOKEN` in config.py

### Knowledge Base Setup

1. In Open WebUI, go to Workspace → Knowledge
2. Create or select your datacenter knowledge base
3. Upload technical documentation
4. Copy the knowledge base ID to `KNOWLEDGE_ID` in config.py

### Speech-to-Text Options

**Option 1: OpenAI Whisper** (Recommended)
```python
USE_LOCAL_WHISPER = False
OPENAI_API_KEY = "your-api-key"
```

**Option 2: Groq** (Faster/Cheaper)
```python
USE_GROQ_STT = True
GROQ_API_KEY = "your-api-key"
```

**Option 3: Local Whisper** (Offline/Free)
```python
USE_LOCAL_WHISPER = True
LOCAL_WHISPER_MODEL = "base"
```

### Text-to-Speech (Optional)

Enable spoken responses:
```python
USE_TTS = True
TOTALGPT_API_KEY = "your-api-key"
TTS_VOICE = "af_bella"
```

## Architecture

```
┌─────────────────┐
│   Technician    │
│  (Smart Glasses)│
└────────┬────────┘
         │
         │ Voice Command
         │ Screenshot
         ▼
┌─────────────────┐
│  BioBot Client  │
│    (macOS)      │
│                 │
│  • Audio Record │
│  • Screenshot   │
│  • STT          │
│  • TTS          │
└────────┬────────┘
         │
         │ API Call
         ▼
┌─────────────────┐
│   Open WebUI    │
│                 │
│  • LLM Models   │
│  • RAG/KB       │
│  • Chat History │
└────────┬────────┘
         │
         │ Knowledge Retrieval
         ▼
┌─────────────────┐
│  Knowledge Base │
│                 │
│  • Datacenter   │
│    Documentation│
└─────────────────┘
```

## API Integration

BioBot integrates with Open WebUI using the following endpoints:

- `POST /api/chat/completions` - Send queries to models
- `POST /api/v1/files/` - Upload screenshots
- `GET /api/v1/chats/{id}` - Retrieve chat history
- `POST /api/v1/chats/{id}` - Save messages with attachments
- `GET /api/models` - Fetch available models

## Troubleshooting


### Connection Issues

**"Could not connect to Open WebUI"**
- Verify Open WebUI is running
- Check `OPENWEBUI_URL` in config.py
- Validate `OPENWEBUI_TOKEN` is correct
- Test: `curl http://YOUR_SERVER_URL/api/config`

### Model Issues

**"Model not found"**
- Verify model exists: `python list_models.py`
- Check model name in `DEFAULT_MODEL`
- Ensure TotalGPT or Ollama endpoint is configured in Open WebUI

### Audio Issues

**"No audio recorded"**
- Grant microphone permissions to Terminal
- Check audio input device in System Preferences
- Adjust `AUDIO_SAMPLE_RATE` if needed

### Screenshot Issues

**"Screenshot capture failed"**
- Grant screen recording permissions to Terminal
- Verify focused window exists
- Falls back to full screen automatically

## Development

### Project Structure

```
biobot/
├── biobot-client/          # Main voice client
│   ├── biobot_voice.py     # Core client implementation
│   ├── config.py.example   # Configuration template
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Client-specific documentation
├── biobot-api/            # API services (future)
├── shared/                # Shared utilities (future)
└── docs/                  # Additional documentation
```

### Running Tests

```bash
cd biobot-client
python test_openwebui.py  # Test Open WebUI connection
python list_models.py     # List available models
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Open WebUI](https://github.com/open-webui/open-webui) for the LLM interface
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [TotalGPT](https://totalgpt.ai) for API services

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/Braryai/biobot).


