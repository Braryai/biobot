# 🔄 Migration Guide: macPerplex → BioBot

This guide helps you transition from the old `macPerplex.py` script (Perplexity-based) to the new `biobot_voice.py` (Open WebUI-based).

## What Changed?

### ✅ What Stays the Same

- **User experience**: Same push-to-talk interaction
- **Keyboard shortcuts**: Same default keys (Right Cmd, Right Shift)
- **Audio recording**: Same audio capture mechanism
- **Screenshot capture**: Same smart window capture
- **Audio transcription**: Same Whisper API integration

### 🔄 What's Different

| Old (macPerplex) | New (BioBot) |
|------------------|--------------|
| Perplexity.ai via browser | Open WebUI via HTTP API |
| Selenium automation | Direct API calls (httpx) |
| Chrome debugging required | No browser needed |
| Web interface responses | Terminal output |
| No knowledge base | RAG with datacenter docs |

### 🚀 New Features

- ✅ **RAG-powered responses** from your datacenter knowledge base
- ✅ **No browser automation** - cleaner, faster, more reliable
- ✅ **Citations included** - know where info comes from
- ✅ **Vision model support** - analyze screenshots properly
- ✅ **Groq support** - faster/cheaper transcription option
- ✅ **Better error handling** - clearer error messages

## Migration Steps

### Step 1: Backup Current Setup

```bash
# Navigate to BioBot directory
cd /Users/svak/Downloads/BioBot

# Backup your current config (if you have one)
cp config.py config.py.backup 2>/dev/null || echo "No config.py to backup"
```

### Step 2: Set Up New Client

```bash
# Navigate to new client directory
cd biobot-client

# Install dependencies (removes Selenium, adds httpx)
pip install -r requirements.txt

# Create new configuration
cp config.py.example config.py
```

### Step 3: Transfer Settings

If you have existing settings from the old `config.py`, transfer them:

**From old config.py:**
```python
OPENAI_API_KEY = "sk-xxxxx..."
OPENAI_STT_MODEL = "whisper-1"
TRIGGER_KEY_WITH_SCREENSHOT = 'cmd_r'
TRIGGER_KEY_AUDIO_ONLY = 'shift_r'
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
MAX_RECORDING_DURATION = 60
```

**To new config.py** (copy these values, plus add new ones):
```python
# Keep your existing values
OPENAI_API_KEY = "sk-xxxxx..."  # Same
OPENAI_STT_MODEL = "whisper-1"   # Same
TRIGGER_KEY_WITH_SCREENSHOT = 'cmd_r'  # Same
TRIGGER_KEY_AUDIO_ONLY = 'shift_r'     # Same
AUDIO_SAMPLE_RATE = 16000  # Same
AUDIO_CHANNELS = 1         # Same
MAX_RECORDING_DURATION = 60  # Same

# Add new Open WebUI settings
OPENWEBUI_URL = "http://YOUR_SERVER_URL"
OPENWEBUI_TOKEN = "sk-xxxxx..."  # Get from Open WebUI
KNOWLEDGE_ID = "xxxxx..."        # Get from Open WebUI
DEFAULT_MODEL = "llama3.1:8b"

# Optional: Use Groq for faster/cheaper STT
USE_GROQ_STT = False
GROQ_API_KEY = ""
```

### Step 4: Get Open WebUI Credentials

#### Get API Token

1. Open http://YOUR_SERVER_URL in browser
2. Click **Settings** (gear icon)
3. Go to **Account** tab
4. Click **API Keys**
5. Click **Create new secret key**
6. Copy the token (starts with `sk-`)
7. Paste into `OPENWEBUI_TOKEN` in config.py

#### Get Knowledge Base ID

1. In Open WebUI, click **Workspace** (sidebar)
2. Click **Knowledge**
3. You should see your datacenter knowledge base
4. Click on it
5. Look at the URL or click settings
6. Copy the ID (UUID format like `abc123-def456-...`)
7. Paste into `KNOWLEDGE_ID` in config.py

### Step 5: Test Connection

```bash
# Test that Open WebUI is accessible
curl http://YOUR_SERVER_URL/api/config

# Should return JSON config
```

### Step 6: Run New Client

```bash
# Make sure you're in biobot-client directory
cd /Users/svak/Downloads/BioBot/biobot-client

# Run the new client
python biobot_voice.py
```

You should see:
```
============================================================
🤖 BioBot Voice Client - Datacenter AI Assistant
============================================================

🔗 Testing connection to Open WebUI...
   URL: http://YOUR_SERVER_URL
✓ Connected to Open WebUI successfully!

============================================================
✅ READY! Two modes:
...
```

### Step 7: Test Both Modes

**Test Audio-Only Mode:**
1. Press and hold Right Shift
2. Say: "What is VLAN 200 used for?"
3. Release
4. Check response includes datacenter info with citations

**Test Audio+Screenshot Mode:**
1. Open a terminal or document
2. Press and hold Right Command
3. Say: "What am I looking at?"
4. Release
5. Check response analyzes the screenshot

## Troubleshooting Migration

### Issue: "Could not connect to Open WebUI"

**Solutions:**
```bash
# Check if Open WebUI is running
curl http://YOUR_SERVER_URL

# If not accessible, check:
# 1. Is the server running?
# 2. Is the URL correct?
# 3. Are you on the right network?
```

### Issue: "HTTP Error 401"

**Solution:** API token is invalid or missing
```bash
# Regenerate token in Open WebUI:
# Settings → Account → API Keys → Create new secret key
```

### Issue: "No responses from knowledge base"

**Solutions:**
1. Verify `KNOWLEDGE_ID` is set correctly
2. Check knowledge base has documents in Open WebUI
3. Test query directly in Open WebUI web interface first

### Issue: Import errors or missing modules

**Solutions:**
```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# If specific module fails (e.g., httpx):
pip install httpx
```

### Issue: Same errors as old script

**Remember:** You don't need Chrome anymore!
- No Chrome debugging setup needed
- No Selenium dependencies
- No browser navigation

## Comparison Table

| Feature | macPerplex (Old) | BioBot (New) |
|---------|------------------|--------------|
| **Backend** | Perplexity.ai | Open WebUI |
| **Connection** | Selenium + Chrome | HTTP API |
| **Setup** | Complex (Chrome debugging) | Simple (just config) |
| **Speed** | Slower (browser) | Faster (API) |
| **Knowledge** | Perplexity's general knowledge | Your datacenter docs |
| **Citations** | Sometimes | Always |
| **Offline** | No | Possible (with local models) |
| **Reliability** | Browser-dependent | More stable |
| **Maintenance** | Higher (Selenium fragile) | Lower (API stable) |

## What to Do with Old Script

### Keep for Reference

```bash
# Old script is still in the root directory
cd /Users/svak/Downloads/BioBot
ls -l macPerplex.py

# It's preserved for reference but not actively used
```

### Optional: Archive It

```bash
# Create archive directory
mkdir -p archive

# Move old files
mv macPerplex.py archive/
mv config.py.example archive/ 2>/dev/null || true
mv requirements.txt archive/requirements-old.txt 2>/dev/null || true
```

## Benefits of Migration

1. **No browser needed** - More reliable, less overhead
2. **Faster responses** - Direct API calls
3. **Better accuracy** - Uses your datacenter documentation
4. **Citations** - Know where information comes from
5. **Vision support** - Better screenshot analysis with vision models
6. **Easier maintenance** - No Selenium to break
7. **More control** - Self-hosted Open WebUI
8. **Privacy** - Data stays on your infrastructure

## Rollback Plan

If you need to go back to the old system:

```bash
# The old script is still available
cd /Users/svak/Downloads/BioBot
python macPerplex.py

# You'll need to:
# 1. Start Chrome in debug mode again
# 2. Navigate to perplexity.ai
# 3. Run the old script
```

## Getting Help

If you encounter issues during migration:

1. Check `biobot-client/README.md` for detailed troubleshooting
2. Verify Open WebUI is working via web interface first
3. Test individual components (audio, screenshot, API)
4. Check logs for detailed error messages

## Next Steps After Migration

1. **Test thoroughly** - Try various queries
2. **Customize shortcuts** - Adjust to your preferences
3. **Add more docs** - Expand knowledge base
4. **Try Groq** - Potentially faster/cheaper transcription
5. **Explore vision models** - Better screenshot analysis

---

Welcome to BioBot! 🤖🎤
