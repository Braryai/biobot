# 🤖 BioBot - Datacenter Voice Assistant

A complete voice-controlled AI assistant system for datacenter technicians using smart glasses or mobile devices. BioBot enables hands-free access to technical documentation through Open WebUI with RAG (Retrieval-Augmented Generation).

## 🎯 Project Overview

BioBot allows datacenter technicians to:
- Ask questions hands-free while working with equipment
- Get instant answers from datacenter documentation
- Include screenshots of equipment for visual context
- Receive accurate responses with source citations

### Example Use Case

**Technician:** "What cable color should I use for VLAN 200?"

**BioBot:** "Yellow cable for VLAN 200 (Database/Storage) [1]"

## 🏗️ Project Structure

```
BioBot/
├── biobot-client/          # Voice client (macOS)
│   ├── biobot_voice.py     # Main voice client script
│   ├── config.py.example   # Configuration template
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Client documentation
│
├── biobot-api/            # API services (future expansion)
│   └── .env.example       # API environment variables
│
├── docs/                  # Datacenter documentation
│   └── (your documentation files)
│
├── shared/                # Shared utilities (future)
│
├── macPerplex.py          # Original script (legacy)
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Open WebUI** instance running at `http://216.81.245.140:8080`
2. **Python 3.11+** installed on macOS
3. **Speech-to-Text API key** - Choose one:
   - OpenAI Whisper API (recommended), OR
   - Groq API (faster/cheaper alternative)
4. **Microphone and screen recording permissions** granted

**Optional:**
- TotalGPT API key for Text-to-Speech (spoken responses)

### Installation

1. **Navigate to the voice client:**
   ```bash
   cd biobot-client
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure settings:**
   ```bash
   cp config.py.example config.py
   nano config.py  # Edit with your credentials
   ```

   Update these values:
   - `OPENWEBUI_URL` = `"http://216.81.245.140:8080"`
   - `OPENWEBUI_TOKEN` = Your API token from Open WebUI
   - `KNOWLEDGE_ID` = Your Knowledge Base ID
   - `OPENAI_API_KEY` = Your OpenAI API key (for STT only!)

   **Note:** OpenAI is only used for Speech-to-Text (voice→text).
   The LLM (llama3.1:8b) runs on your Open WebUI instance!

4. **Run BioBot:**
   ```bash
   python biobot_voice.py
   ```

### Getting API Credentials

#### Open WebUI Token
1. Open http://216.81.245.140:8080
2. Settings → Account → API Keys
3. Create new secret key → Copy

#### Knowledge Base ID
1. Workspace → Knowledge
2. Click your datacenter knowledge base
3. Copy the ID from URL or settings

#### OpenAI API Key
1. Visit https://platform.openai.com/api-keys
2. Create new secret key → Copy

**What it's used for:** Speech-to-Text ONLY (converting voice to text)
**NOT used for:** LLM responses (that's Open WebUI's job!)

**Alternative:** Use Groq instead (faster/cheaper) by setting `USE_GROQ_STT = True`

## 📋 Features

### Current Features ✅
- ✅ Push-to-talk audio recording with visual feedback
- ✅ Two modes: Audio+Screenshot and Audio-only
- ✅ OpenAI Whisper transcription (with Groq support)
- ✅ Direct Open WebUI API integration (no browser automation!)
- ✅ RAG-powered responses from knowledge base
- ✅ Vision model support for screenshot analysis
- ✅ Automatic cleanup of temporary files
- ✅ Configurable keyboard shortcuts
- ✅ Clean terminal output with citations
- ✅ Text-to-Speech responses (optional, via TotalGPT)

## 🎤 Usage

### Two Operating Modes

**Mode 1: Audio + Screenshot** (Right Command key)
- Hold Right Command (⌘)
- Speak your question
- Release
- BioBot captures screenshot and responds


### Example Session

```
🎯 Cmd (Right) PRESSED - Recording with screenshot...
🎤 Recording with screenshot... (release key to stop)
   Audio levels: ███▓▓.████▓▓▓...
✓ Recording stopped
✓ Transcription: "What's the IP range for the management network?"

============================================================
🤖 BIOBOT RESPONSE:
============================================================
The management network uses IP range 10.0.1.0/24 [1]

This range is reserved for out-of-band management interfaces 
(iDRAC, iLO, IPMI) and should not be used for production 
traffic. Gateway is 10.0.1.1 [2].

[1] Network_Configuration_Guide.pdf
[2] IP_Address_Schema.pdf
============================================================
```

## 🔧 Configuration

See `biobot-client/README.md` for detailed configuration options.

**Essential Settings:**
```python
# Open WebUI (your LLM server)
OPENWEBUI_URL = "http://216.81.245.140:8080"
OPENWEBUI_TOKEN = "sk-xxxxx..."
KNOWLEDGE_ID = "your-kb-id"
DEFAULT_MODEL = "llama3.1:8b"

# Speech-to-Text (voice → text)
OPENAI_API_KEY = "sk-xxxxx..."  # For Whisper STT only!
USE_GROQ_STT = False  # Set True for Groq (faster/cheaper)

# Text-to-Speech (optional, speaks responses)
USE_TTS = False  # Set True to enable
TOTALGPT_API_KEY = "sk-dOwBzdjuw0OWIgYAyytZoA"
```

## 🛠️ Troubleshooting

**Cannot connect to Open WebUI:**
- Verify server is running: `curl http://216.81.245.140:8080`
- Check URL and token in config.py

**No audio recording:**
- Grant microphone permissions to Terminal
- System Settings → Privacy → Microphone

**Screenshot not captured:**
- Grant screen recording permissions to Terminal  
- System Settings → Privacy → Screen Recording

**See `biobot-client/README.md` for complete troubleshooting guide.**

## 📚 Documentation

- **Client Guide:** `biobot-client/README.md` - Complete usage and troubleshooting
- **Configuration:** `biobot-client/config.py.example` - All settings explained
- **Legacy Script:** `macPerplex.py` - Original Perplexity version (deprecated)

## 🏢 System Architecture

```
┌─────────────────┐
│  Technician     │
│  (macOS)        │
└────────┬────────┘
         │ Voice + Screenshot
         ▼
┌─────────────────┐
│  BioBot Client  │
│  biobot_voice.py│
└────────┬────────┘
         │ Whisper API
         ▼
┌─────────────────┐
│  OpenAI/Groq    │
│  (STT Service)  │
└─────────────────┘
         │ Transcribed Text + Image
         ▼
┌─────────────────┐
│  Open WebUI     │
│  216.81.245     │
│  .140:8080      │
└────────┬────────┘
         │ RAG Query
         ▼
┌─────────────────┐
│  Knowledge Base │
│  (Datacenter    │
│   Documentation)│
└────────┬────────┘
         │ Response + Citations
         ▼
┌─────────────────┐
│  Technician     │
│  (Terminal)     │
└─────────────────┘
```

### Technology Stack

**Client:** Python 3.11+, pynput, sounddevice, OpenAI SDK, httpx, PyObjC  
**Server:** Open WebUI, Ollama (llama3.1:8b), ChromaDB, RAG  
**External Services:**
- OpenAI Whisper API or Groq (Speech-to-Text only)
- TotalGPT API (Text-to-Speech, optional)

## 🔐 Security

- Never commit `config.py` with real credentials
- Temporary files auto-deleted after processing
- Use HTTPS for Open WebUI in production
- Regenerate tokens if compromised

## 📖 Project Roadmap

**✅ Phase 1: Core (Current)**
- Voice input with push-to-talk
- Screenshot capture  
- Open WebUI integration
- RAG responses

**🔄 Phase 2: Enhanced UX (Next)**
- Text-to-speech responses
- Conversation history
- Improved error handling

**📅 Phase 3: Mobile (Future)**
- Android/iOS apps
- Smart glasses integration

## 🤝 Contributing

Internal project for datacenter operations. For contributions:
1. Create feature branch
2. Test thoroughly
3. Update documentation
4. Submit for review

## 📄 License

See LICENSE file for details.

---

**BioBot** - Making datacenter operations hands-free 🎤🤖


