# 🚀 QUICKSTART - BioBot Voice Client

Get BioBot running in 5 minutes!

## Prerequisites

✅ macOS computer  
✅ Python 3.11+ installed  
✅ Open WebUI running at your server  
✅ OpenAI API key (for Whisper transcription)  

## Installation (2 minutes)

```bash
# 1. Navigate to client directory
cd biobot-client

# 2. Run setup script (installs dependencies & creates config)
./setup.sh

# 3. Edit configuration with your credentials
nano config.py
```

## Configuration (2 minutes)

Edit `config.py` and set these 4 values:

```python
OPENWEBUI_URL = "http://YOUR_SERVER_URL"  # Set your server URL
OPENWEBUI_TOKEN = "sk-xxxxx..."  # Get from Open WebUI
KNOWLEDGE_ID = "xxxxx..."        # Get from Open WebUI  
OPENAI_API_KEY = "sk-xxxxx..."   # Get from OpenAI
```

### Get API Credentials

**Open WebUI Token:**
1. Open http://YOUR_SERVER_URL
2. Settings → Account → API Keys → Create new secret key

**Knowledge Base ID:**
1. Workspace → Knowledge → (click your datacenter KB)
2. Copy the ID

**OpenAI Key:**
1. https://platform.openai.com/api-keys → Create new secret key

## Run BioBot (1 minute)

```bash
python3 biobot_voice.py
```

You should see:
```
============================================================
🤖 BioBot Voice Client - Datacenter AI Assistant
============================================================

🔗 Testing connection to Open WebUI...
✓ Connected to Open WebUI successfully!

✅ READY! Two modes:
   🖼️  Cmd (Right) - Audio + Screenshot
   🎤 Shift (Right) - Audio Only
============================================================
```

## Test It!

**Audio Only:**
1. Hold Right Shift key
2. Say: "What is VLAN 200 used for?"
3. Release
4. See response with citations!

**Audio + Screenshot:**
1. Open a terminal or document
2. Hold Right Command key
3. Say: "What am I looking at?"
4. Release
5. See response analyzing the screenshot!

## Troubleshooting

**Connection error?**
→ Check Open WebUI is running: `curl http://YOUR_SERVER_URL`

**No audio?**
→ Grant microphone permissions: System Settings → Privacy → Microphone

**No screenshot?**
→ Grant screen recording: System Settings → Privacy → Screen Recording

**API error?**
→ Check your OPENWEBUI_TOKEN is correct

## Full Documentation

- **Complete Guide:** `README.md`
- **Migration from macPerplex:** `../MIGRATION.md`
- **Troubleshooting:** See README.md

---

**That's it! You're ready to use BioBot! 🤖🎤**
