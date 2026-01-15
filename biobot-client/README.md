# 🤖 BioBot Voice Client

Voice-controlled AI assistant for datacenter technicians using Open WebUI with RAG-enabled knowledge base.

## Overview

BioBot Voice Client enables hands-free interaction with your datacenter documentation through:
- **Push-to-talk audio recording** with visual feedback
- **Optional screenshot capture** of equipment/screens
- **Automatic speech-to-text** transcription using OpenAI Whisper or Groq
- **RAG-powered responses** from your datacenter knowledge base via Open WebUI
- **Clean terminal output** with responses including citations

## Features

### Two Operating Modes
1. **Audio + Screenshot Mode** (Right Command key)
   - Captures audio question
   - Captures screenshot of current window
   - Sends both to Open WebUI for context-aware responses

2. **Audio-Only Mode** (Right Shift key)
   - Captures audio question only
   - Sends query without screenshot

### Key Capabilities
- ✅ Push-to-talk recording with real-time volume indicators
- ✅ Smart window capture (focused window or full screen fallback)
- ✅ Multiple STT options (OpenAI Whisper or Groq)
- ✅ Direct API integration with Open WebUI (no browser automation)
- ✅ Knowledge Base integration for RAG queries
- ✅ Support for vision models with screenshot analysis
- ✅ Automatic cleanup of temporary files
- ✅ Configurable keyboard shortcuts

## Prerequisites

### System Requirements
- **OS**: macOS (uses macOS-specific screenshot tools)
- **Python**: 3.11 or higher
- **Open WebUI**: Running instance with API access

### Required Services
1. **Open WebUI** running at `http://216.81.245.140:8080` (provides the LLM)
2. **Speech-to-Text** - Choose one:
   - **Local Whisper** ⭐ RECOMMENDED - Free, offline, private! (just `pip install faster-whisper`)
   - **Groq Whisper API** - Free (for now), requires internet
   - **OpenAI Whisper API** - Paid, reliable
3. **Knowledge Base** configured in Open WebUI with datacenter documentation
4. **Text-to-Speech** (optional) - TotalGPT API for spoken responses

### API Clarification

**What each service does:**
- **Open WebUI** = Your LLM (llama3.1:8b) that answers questions using RAG
- **Local Whisper** = Speech-to-Text running on your Mac (FREE & OFFLINE!) ⭐
- **Groq/OpenAI** = Alternative cloud STT (if you don't want local)
- **TotalGPT** = Text-to-Speech only (optional, speaks responses back to you)

## Installation

### 1. Install Python Dependencies

```bash
cd biobot-client
pip install -r requirements.txt
```

### 2. Configure Settings

Copy the example configuration file and edit with your credentials:

```bash
cp config.py.example config.py
nano config.py  # or use your preferred editor
```

**Required settings to update:**

```python
# Open WebUI Configuration
OPENWEBUI_URL = "http://216.81.245.140:8080"  # Already set correctly
OPENWEBUI_TOKEN = "sk-xxxxx..."  # Your API token from Open WebUI
KNOWLEDGE_ID = "xxxxx-xxxxx-xxxxx"  # Your Knowledge Base ID

# AI Model
DEFAULT_MODEL = "llama3.1:8b"  # Or your preferred model

# Speech-to-Text
OPENAI_API_KEY = "sk-xxxxx..."  # Your OpenAI API key
```

**Optional settings:**
```python
# Use Groq for faster/cheaper transcription (optional)
USE_GROQ_STT = True
GROQ_API_KEY = "gsk_xxxxx..."

# Customize keyboard shortcuts (optional)
TRIGGER_KEY_WITH_SCREENSHOT = 'cmd_r'  # Right Command
TRIGGER_KEY_AUDIO_ONLY = 'shift_r'     # Right Shift
```

### 3. Get API Credentials

#### Open WebUI API Token
1. Open your Open WebUI instance in a browser
2. Go to **Settings** → **Account** → **API Keys**
3. Click **Create new secret key**
4. Copy the token (starts with `sk-`)
5. Paste into `config.py` as `OPENWEBUI_TOKEN`

#### Knowledge Base ID
1. In Open WebUI, go to **Workspace** → **Knowledge**
2. Find your datacenter knowledge base
3. Click on it and look at the URL or settings
4. Copy the ID (UUID format)
5. Paste into `config.py` as `KNOWLEDGE_ID`

#### Speech-to-Text Setup

**RECOMMENDED: Local Whisper (Free & Offline!)** ⭐

No API key needed! Just install:
```bash
pip install faster-whisper
```

Then in config.py:
```python
USE_LOCAL_WHISPER = True
LOCAL_WHISPER_MODEL = "base"  # or "small" for better accuracy
```

**See `../LOCAL_STT_GUIDE.md` for complete guide and model comparison!**

**Alternative 1: Groq API (Free cloud option)**
1. Go to https://console.groq.com/keys
2. Create API key
3. Set `USE_LOCAL_WHISPER = False` in config.py
4. Set `USE_GROQ_STT = True`
5. Paste key as `GROQ_API_KEY`

**Alternative 2: OpenAI API (Paid cloud option)**
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Set `USE_LOCAL_WHISPER = False` in config.py
4. Paste key as `OPENAI_API_KEY`

#### TotalGPT API Key (optional, for Text-to-Speech)
1. Your key is already configured: `sk-dOwBzdjuw0OWIgYAyytZoA`
2. Set `USE_TTS = True` in config.py to enable spoken responses
3. Customize voice with `TTS_VOICE` (e.g., "af_bella", "am_adam")

## Usage

### Starting BioBot

```bash
cd biobot-client
python biobot_voice.py
```

You should see:
```
============================================================
🤖 BioBot Voice Client - Datacenter AI Assistant
============================================================

🔗 Testing connection to Open WebUI...
   URL: http://216.81.245.140:8080
✓ Connected to Open WebUI successfully!

============================================================
✅ READY! Two modes:

   🖼️  Cmd (Right) - Audio + Screenshot
      Hold, speak, release → sends with image

   🎤 Shift (Right) - Audio Only
      Hold, speak, release → sends without image

   💡 Using modifier keys prevents beeping!
   Press Ctrl+C to exit
============================================================
```

### Recording a Query

#### With Screenshot (Right Command Key)
1. **Press and hold** Right Command key (⌘)
2. **Speak** your question (you'll see audio level indicators: `█▓.`)
3. **Release** the key when done
4. BioBot will:
   - Stop recording
   - Capture a screenshot
   - Transcribe your audio
   - Send to Open WebUI
   - Display the response

#### Without Screenshot (Right Shift Key)
1. **Press and hold** Right Shift key (⇧)
2. **Speak** your question
3. **Release** the key
4. BioBot processes without screenshot

### Example Session

```
============================================================
🎯 Cmd (Right) PRESSED - Recording with screenshot...
============================================================
🎤 Recording with screenshot... (release key to stop)
   Audio levels: ███▓▓.████▓▓▓...
✓ Recording stopped
✓ Audio saved: /tmp/biobot_audio_20260104_143052.wav (3.2 seconds)

============================================================
🎯 PROCESSING QUERY...
============================================================
🔄 Transcribing audio...
   Using OpenAI Whisper...
✓ Transcription: "What cable color should I use for VLAN 200?"

📸 Capturing screenshot...
   Frontmost app: Terminal (PID: 12345)
   Window ID: 54321
   Window bounds: 1920x1080
✓ Captured focused window!
  Dimensions: 1920 x 1080 pixels
✓ Captured screenshot: /tmp/biobot_screenshot_20260104_143052.png
  File size: 0.85 MB (891,234 bytes)

🔄 Sending query to Open WebUI...
   Using Knowledge Base: abc123-def456-ghi789
   API endpoint: http://216.81.245.140:8080/api/chat/completions
✓ Received response from Open WebUI

============================================================
🤖 BIOBOT RESPONSE:
============================================================
Yellow cable for VLAN 200 (Database/Storage) [1]

According to the datacenter cabling standards, VLAN 200 is 
designated for database and storage traffic, which uses yellow 
cables for easy identification. This is part of the color-coding 
system designed to help technicians quickly identify cable 
purposes during maintenance and troubleshooting [2].

[1] Cabling_Standards.pdf
[2] Network_Configuration_Guide.pdf
============================================================

✅ Query complete!
============================================================
⌨️  Ready for next query (Right Cmd = w/ screenshot, Right Shift = audio only)...

🗑️  Cleaned up audio file
🗑️  Cleaned up screenshot file
```

## Keyboard Shortcuts

Default shortcuts (can be customized in `config.py`):

| Key | Action |
|-----|--------|
| **Right Command (⌘)** | Hold to record audio + capture screenshot |
| **Right Shift (⇧)** | Hold to record audio only (no screenshot) |
| **Ctrl+C** | Exit BioBot |

### Available Key Options

You can customize shortcuts in `config.py`:
- `cmd_r` - Right Command
- `cmd` - Left Command
- `shift_r` - Right Shift
- `shift` - Left Shift
- `alt_r` - Right Option/Alt
- `alt` - Left Option/Alt
- `ctrl_r` - Right Control
- `ctrl` - Left Control

**Why modifier keys?** They prevent the annoying macOS beep sound and don't interfere with other applications.

## Troubleshooting

### Connection Issues

**Error: Could not connect to Open WebUI**
```
❌ Could not connect to Open WebUI: ...
```

**Solutions:**
1. Verify Open WebUI is running: `curl http://216.81.245.140:8080/api/config`
2. Check the URL in `config.py` matches your Open WebUI instance
3. Verify your API token is valid (regenerate if needed)
4. Check firewall/network settings

### Authentication Issues

**Error: HTTP Error 401 or 403**

**Solutions:**
1. Regenerate API token in Open WebUI (Settings → Account → API Keys)
2. Ensure token is copied correctly to `config.py` (no extra spaces)
3. Verify token has not expired

### Audio Recording Issues

**Error: Error starting recording**

**Solutions:**
1. Check microphone permissions in macOS System Settings
2. Grant Terminal (or your terminal app) microphone access
3. Test microphone with another app to verify it works
4. Try a different audio input device

**No audio levels showing (all dots)**

**Solutions:**
1. Speak louder or move closer to microphone
2. Check system audio input level in System Settings
3. Select correct input device in System Settings → Sound

### Screenshot Issues

**Error: Screenshot file was not created**

**Solutions:**
1. Check Screen Recording permissions in System Settings → Privacy
2. Grant Terminal (or your terminal app) screen recording access
3. Restart terminal after granting permissions

### Transcription Issues

**Error: Error transcribing audio**

**Solutions:**
1. Verify OpenAI API key is correct
2. Check you have credits in your OpenAI account
3. Try using Groq instead (set `USE_GROQ_STT = True`)
4. Ensure audio file was created successfully

**Poor transcription quality**

**Solutions:**
1. Speak more clearly and slowly
2. Reduce background noise
3. Move closer to microphone
4. Try a better microphone

### Knowledge Base Issues

**Responses don't include datacenter information**

**Solutions:**
1. Verify `KNOWLEDGE_ID` is set correctly in `config.py`
2. Check knowledge base is properly configured in Open WebUI
3. Ensure documents are uploaded and indexed
4. Test RAG directly in Open WebUI interface first

### Python Dependency Issues

**Error: ImportError or ModuleNotFoundError**

**Solutions:**
```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# For PyObjC issues on macOS:
pip install --upgrade pyobjc-framework-Quartz pyobjc-framework-Cocoa
```

## Advanced Configuration

### Using Groq for Faster Transcription

Groq offers faster and cheaper Whisper transcription:

1. Get API key from https://console.groq.com/keys
2. Update `config.py`:
```python
USE_GROQ_STT = True
GROQ_API_KEY = "gsk_xxxxx..."
```

### Custom Model Selection

To use a different model from your Open WebUI instance:

```python
# Use a vision-capable model for better screenshot analysis
DEFAULT_MODEL = "llama3.2-vision:11b"

# Or use a larger model for better responses
DEFAULT_MODEL = "llama3.1:70b"
```

### Enable Text-to-Speech

Make BioBot speak responses back to you:

```python
# Enable TTS
USE_TTS = True

# Configure voice and speed
TTS_VOICE = "af_bella"  # Female voice (Bella)
# Other options: "am_adam" (male), "af_sarah", "am_michael", etc.

TTS_SPEED = 1.0  # Normal speed (0.1-4.0)

TTS_LANG_CODE = "a"  # English
# Other options: "e" (Spanish), "f" (French), "h" (Hindi), etc.
```

**Available Voices:**
- Female: `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- Male: `am_adam`, `am_michael`, `am_charlie`
- And 50+ more at https://api.totalgpt.ai/docs

### Adjusting Audio Settings

```python
# Higher quality audio (slower processing)
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2  # Stereo

# Longer recordings
MAX_RECORDING_DURATION = 120  # 2 minutes
```

## API Reference

### Open WebUI API Endpoint

The client uses the following endpoint:
```
POST http://216.81.245.140:8080/api/chat/completions
```

**Request format:**
```json
{
  "model": "llama3.1:8b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What cable color for VLAN 200?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,..."
          }
        }
      ]
    }
  ],
  "files": [
    {
      "type": "collection",
      "id": "knowledge-base-id"
    }
  ],
  "stream": false
}
```

## Performance Tips

1. **Use Groq for transcription** - Faster and cheaper than OpenAI
2. **Audio-only mode** - Skip screenshot when not needed for faster responses
3. **Smaller models** - `llama3.1:8b` is faster than `llama3.1:70b`
4. **Clean knowledge base** - Keep only relevant documents indexed
5. **Local network** - Run Open WebUI on local network for lower latency

## Security Notes

⚠️ **Important Security Considerations:**

1. **API Keys**: Never commit `config.py` with real keys to version control
2. **Screenshots**: May capture sensitive information - temporary files are auto-deleted
3. **Audio**: Recordings are temporary and auto-deleted after transcription
4. **Network**: Consider using HTTPS for Open WebUI in production
5. **Tokens**: Regenerate API tokens if compromised

## Support & Feedback

For issues, questions, or feature requests:
1. Check this README's troubleshooting section
2. Verify Open WebUI is working correctly via web interface
3. Test individual components (audio recording, screenshot, API connection)
4. Check logs for detailed error messages

## License

See LICENSE file in the project root.
