# 🔌 API Usage Clarification

## Understanding BioBot's API Architecture

BioBot uses **three separate APIs** for different purposes. Here's what each one does:

---

## 🎯 API Roles

### 1️⃣ Open WebUI (YOUR_SERVER_URL)
**Role:** Your AI Brain 🧠

**What it does:**
- Runs the LLM (llama3.1:8b)
- Provides RAG (Retrieval-Augmented Generation)
- Queries your datacenter knowledge base
- Generates intelligent responses
- Analyzes screenshots with vision models

**What it provides:**
- AI-powered answers to your questions
- Source citations from documentation
- Context-aware responses using RAG

**API Key:** `OPENWEBUI_TOKEN`  
**Endpoint:** `http://YOUR_SERVER_URL/api/chat/completions`

---

### 2️⃣ OpenAI Whisper (or Groq)
**Role:** Your Ears 👂 (Speech-to-Text)

**What it does:**
- Converts your voice recordings to text
- Transcribes audio files
- **ONLY does STT - nothing else!**

**What it DOES NOT do:**
- ❌ Does NOT generate AI responses
- ❌ Does NOT query your knowledge base
- ❌ Does NOT analyze images
- ❌ Does NOT run the LLM

**API Key:** `OPENAI_API_KEY` or `GROQ_API_KEY`  
**Endpoint:** OpenAI Whisper API or Groq API  
**When used:** Every time you speak into the microphone

**Why you need it:**
- Open WebUI doesn't provide STT
- You need to convert speech → text before sending to the LLM
- Groq is a faster/cheaper alternative to OpenAI

---

### 3️⃣ TotalGPT (api.totalgpt.ai)
**Role:** Your Mouth 🗣️ (Text-to-Speech) - OPTIONAL

**What it does:**
- Converts text responses to speech
- Plays audio responses back to you
- Makes BioBot truly hands-free

**What it DOES NOT do:**
- ❌ Does NOT generate AI responses
- ❌ Does NOT transcribe audio
- ❌ Does NOT run the LLM

**API Key:** `TOTALGPT_API_KEY` = `your_totalgpt_api_key_here`  
**Endpoint:** `https://api.totalgpt.ai/v1/audio/speech`  
**When used:** After receiving response from Open WebUI (if `USE_TTS = True`)

**Why it's optional:**
- You can read responses on screen without TTS
- Enable it for truly hands-free operation
- Great for datacenter work where you can't look at screen

---

## 🔄 Complete Workflow

Here's what happens when you use BioBot:

```
1. 🎤 YOU SPEAK
   "What cable color for VLAN 200?"
   
2. 📼 BIOBOT RECORDS
   Saves audio to /tmp/biobot_audio_*.wav
   
3. 👂 OPENAI/GROQ (STT)
   Converts speech → text
   Input: audio file
   Output: "What cable color for VLAN 200?"
   
4. 🧠 OPEN WEBUI (LLM + RAG)
   Processes text query with knowledge base
   Input: text query + knowledge base ID
   Output: "Yellow cable for VLAN 200 (Database/Storage) [1]"
   
5. 📺 TERMINAL DISPLAY
   Shows response with citations
   
6. 🗣️ TOTALGPT (TTS) - Optional
   Converts response → speech
   Input: text response
   Output: spoken audio
   Plays: "Yellow cable for VLAN 200..."
```

---

## 💰 Cost Breakdown

**Per Query Costs (approximate):**

| Service | Cost | What You Pay For |
|---------|------|------------------|
| **Open WebUI** | FREE (self-hosted) | Electricity to run server |
| **OpenAI Whisper** | ~$0.006 per minute | Speech-to-Text |
| **Groq Whisper** | FREE (for now) | Speech-to-Text |
| **TotalGPT TTS** | Variable | Text-to-Speech (check pricing) |

**Recommendation:**
- Use Groq for STT (free/cheap) instead of OpenAI
- Keep TTS optional (only enable when truly hands-free)
- Open WebUI is already self-hosted (no per-query cost!)

---

## ⚙️ Configuration Options

### Option 1: OpenAI STT (Recommended for Quality)
```python
USE_GROQ_STT = False
OPENAI_API_KEY = "sk-xxxxx..."
```

**Pros:** Best transcription quality  
**Cons:** Costs $0.006/minute

### Option 2: Groq STT (Recommended for Cost)
```python
USE_GROQ_STT = True
GROQ_API_KEY = "gsk_xxxxx..."
```

**Pros:** Free/very cheap, faster  
**Cons:** Slightly lower quality (still very good)

### Option 3: Enable TTS
```python
USE_TTS = True
TOTALGPT_API_KEY = "your_totalgpt_api_key_here"
TTS_VOICE = "af_bella"
```

**Pros:** Truly hands-free, hear responses  
**Cons:** Adds cost, takes extra time

---

## 🎯 Why This Architecture?

**Why not use OpenAI for everything?**
- ❌ Can't self-host (closed source)
- ❌ Can't customize knowledge base easily
- ❌ Can't use your own documentation
- ❌ More expensive per query
- ❌ Data leaves your infrastructure

**Why Open WebUI for LLM?**
- ✅ Self-hosted (your infrastructure)
- ✅ Free to run (just server costs)
- ✅ Full control over models
- ✅ Your data stays private
- ✅ Easy RAG integration
- ✅ Can switch models anytime

**Why external APIs for STT/TTS?**
- ✅ Specialized services do it better
- ✅ Don't need to run additional models
- ✅ Can choose best provider for each
- ✅ Easy to switch providers

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                    YOU (Technician)                 │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ 🎤 Voice Recording
                    ▼
┌─────────────────────────────────────────────────────┐
│              BioBot Voice Client                    │
│              (biobot_voice.py)                      │
└────┬──────────────┬──────────────┬──────────────────┘
     │              │              │
     │              │              │
     ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────────┐
│ OpenAI/ │  │   Open   │  │  TotalGPT    │
│  Groq   │  │  WebUI   │  │    (TTS)     │
│  (STT)  │  │  (LLM)   │  │  [Optional]  │
└─────────┘  └──────────┘  └──────────────┘
     │              │              │
     │              │              │
     └──────┬───────┴──────┬───────┘
            │              │
            ▼              ▼
    "text query"    "AI response"
```

---

## 🔐 Security Notes

**API Keys Storage:**
- All keys stored in local `config.py`
- Never committed to version control
- Never sent to wrong services

**Data Privacy:**
| Data Type | Where It Goes |
|-----------|---------------|
| Voice recordings | OpenAI/Groq → deleted after STT |
| Transcribed text | Open WebUI (your server) |
| AI responses | Open WebUI (your server) |
| Screenshots | Open WebUI (your server) |
| Documentation | Open WebUI (your server) |

**What leaves your infrastructure:**
- ✅ Audio files (to STT service, then deleted)
- ✅ TTS requests (if enabled)
- ❌ Your questions stay local (to Open WebUI)
- ❌ AI responses stay local (from Open WebUI)
- ❌ Documentation stays local (in knowledge base)

---

## ❓ FAQ

**Q: Can I run everything locally?**  
A: Almost! You can:
- ✅ Run Open WebUI locally (LLM)
- ✅ Run Whisper locally (requires setup)
- ❌ TTS harder to self-host

**Q: Do I need OpenAI at all?**  
A: Only for STT. Use Groq instead to avoid OpenAI entirely!

**Q: Why not use Open WebUI for STT too?**  
A: Open WebUI doesn't provide STT API (yet). It focuses on LLM/chat.

**Q: Can I use a different TTS service?**  
A: Yes! The code can be modified to use any TTS API. TotalGPT is just the current implementation.

**Q: What if OpenAI/Groq goes down?**  
A: You can't transcribe new voice queries, but you can still type queries directly to Open WebUI's web interface.

**Q: What if Open WebUI goes down?**  
A: BioBot won't work. It's the core AI brain. Make sure your server is reliable!

---

## 🚀 Optimization Tips

**To reduce costs:**
1. Use Groq instead of OpenAI for STT (FREE!)
2. Disable TTS unless truly needed
3. Keep recordings short and clear

**To improve speed:**
1. Use Groq for STT (faster than OpenAI)
2. Use smaller models on Open WebUI (llama3.1:8b vs 70b)
3. Disable TTS during testing

**To improve quality:**
1. Use OpenAI Whisper for best STT accuracy
2. Use vision-capable models for screenshot analysis
3. Enable TTS for better user experience

---

## 📝 Summary

**Remember:**
- **Open WebUI** = Your AI brain (LLM + RAG)
- **OpenAI/Groq** = Speech-to-Text ONLY
- **TotalGPT** = Text-to-Speech (optional)

**You need:**
- ✅ Open WebUI (always)
- ✅ STT API (OpenAI OR Groq)
- ⚪ TTS API (optional)

**Cost optimization:**
- Use Groq for free STT
- Disable TTS unless needed
- Open WebUI is already free (self-hosted)

---

Questions? Check the main README.md or biobot-client/README.md!
