# 🎤 Local Speech-to-Text Guide

## Why Use Local STT?

**Benefits:**
- ✅ **100% Free** - No API costs ever
- ✅ **Offline** - Works without internet
- ✅ **Private** - Audio never leaves your computer
- ✅ **Fast** - No network latency
- ✅ **Unlimited** - No rate limits or quotas

**Trade-offs:**
- ⚠️ Uses CPU/RAM (but modern Macs handle it fine)
- ⚠️ First run downloads model (~150MB-3GB depending on size)
- ⚠️ Slightly slower than cloud APIs (but still acceptable)

---

## 🚀 Quick Setup

### 1. Install faster-whisper

```bash
cd biobot-client
pip install faster-whisper
```

### 2. Enable in config.py

```python
# Enable local Whisper
USE_LOCAL_WHISPER = True

# Choose model size (see below)
LOCAL_WHISPER_MODEL = "base"  # Good default

# Disable API options
USE_GROQ_STT = False
# OPENAI_API_KEY not needed anymore!
```

### 3. Run BioBot

```bash
python biobot_voice.py
```

**First run:** Model will download automatically (takes 1-2 minutes)  
**Subsequent runs:** Uses cached model (instant startup)

---

## 📊 Model Comparison

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| **tiny** | 75 MB | ⚡⚡⚡⚡⚡ | ⭐⭐ | Testing only |
| **base** | 145 MB | ⚡⚡⚡⚡ | ⭐⭐⭐ | **Default choice** ✅ |
| **small** | 466 MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Better accuracy |
| **medium** | 1.5 GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best quality |
| **large-v2** | 3.1 GB | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy |
| **large-v3** | 3.1 GB | ⚡ | ⭐⭐⭐⭐⭐ | Latest & best |

### Recommendations

**For most users:**
```python
LOCAL_WHISPER_MODEL = "base"  # Fast & accurate enough
```

**For better accuracy:**
```python
LOCAL_WHISPER_MODEL = "small"  # Still fast, better results
```

**For maximum quality:**
```python
LOCAL_WHISPER_MODEL = "medium"  # Slower but very accurate
```

**For professional use:**
```python
LOCAL_WHISPER_MODEL = "large-v3"  # Best quality, slower
```

---

## ⚡ Performance Comparison

### Transcription Speed (3-second audio clip)

| Method | Time | Cost | Internet |
|--------|------|------|----------|
| **Local Whisper (tiny)** | ~0.5s | Free | No |
| **Local Whisper (base)** | ~1.0s | Free | No |
| **Local Whisper (small)** | ~2.0s | Free | No |
| **Groq API** | ~0.8s | Free* | Yes |
| **OpenAI API** | ~1.5s | $0.006/min | Yes |

*Free tier may change

### Memory Usage

| Model | RAM Required |
|-------|--------------|
| tiny | ~1 GB |
| base | ~1 GB |
| small | ~2 GB |
| medium | ~5 GB |
| large | ~10 GB |

**Your Mac can easily handle base/small models!**

---

## 🔧 Configuration Examples

### Example 1: Fastest (Offline)
```python
USE_LOCAL_WHISPER = True
LOCAL_WHISPER_MODEL = "tiny"  # Fastest, lower accuracy
```

### Example 2: Balanced (Recommended)
```python
USE_LOCAL_WHISPER = True
LOCAL_WHISPER_MODEL = "base"  # Good speed, good accuracy
```

### Example 3: Best Quality (Offline)
```python
USE_LOCAL_WHISPER = True
LOCAL_WHISPER_MODEL = "small"  # Better accuracy, still fast
```

### Example 4: Fallback Chain
```python
# Try local first, fall back to API if needed
USE_LOCAL_WHISPER = True
LOCAL_WHISPER_MODEL = "base"
USE_GROQ_STT = True  # Fallback if local fails
GROQ_API_KEY = "gsk_xxxxx..."
```

---

## 🎯 How It Works

### Workflow with Local Whisper

```
1. You speak → BioBot records audio
2. Audio saved to /tmp/biobot_audio_*.wav
3. faster-whisper loads model (cached after first run)
4. Model transcribes audio locally on your Mac
5. Text returned → sent to Open WebUI
6. Response from LLM → displayed/spoken
```

**No external API calls for STT!**

---

## 💾 Model Storage

Models are downloaded to:
```
~/.cache/huggingface/hub/
```

**First run:**
- Downloads model from HuggingFace
- Takes 1-5 minutes depending on model size
- Requires internet for download only

**Subsequent runs:**
- Uses cached model
- No internet needed
- Instant loading

**To clear cache:**
```bash
rm -rf ~/.cache/huggingface/hub/
```

---

## 🐛 Troubleshooting

### Issue: "faster-whisper not installed"

**Solution:**
```bash
pip install faster-whisper
```

If that fails:
```bash
pip install --upgrade pip
pip install faster-whisper
```

### Issue: "Model download failed"

**Solution:**
- Check internet connection
- Model downloads from HuggingFace (needs internet first time)
- Retry - sometimes servers are slow

### Issue: "Too slow on my Mac"

**Solutions:**
1. Use smaller model:
   ```python
   LOCAL_WHISPER_MODEL = "tiny"  # or "base"
   ```

2. Or switch to Groq API:
   ```python
   USE_LOCAL_WHISPER = False
   USE_GROQ_STT = True
   ```

### Issue: "Out of memory"

**Solutions:**
1. Use smaller model:
   ```python
   LOCAL_WHISPER_MODEL = "base"  # Instead of large
   ```

2. Close other apps to free RAM

3. Use API instead:
   ```python
   USE_LOCAL_WHISPER = False
   USE_GROQ_STT = True
   ```

### Issue: "Accuracy not good enough"

**Solutions:**
1. Use larger model:
   ```python
   LOCAL_WHISPER_MODEL = "small"  # or "medium"
   ```

2. Speak more clearly and slowly

3. Reduce background noise

4. Use better microphone

---

## 🔄 Switching Between STT Methods

### Local → Groq
```python
USE_LOCAL_WHISPER = False  # Disable local
USE_GROQ_STT = True        # Enable Groq
GROQ_API_KEY = "gsk_xxxxx..."
```

### Local → OpenAI
```python
USE_LOCAL_WHISPER = False  # Disable local
USE_GROQ_STT = False       # Keep disabled
OPENAI_API_KEY = "sk_xxxxx..."
```

### Groq → Local
```python
USE_LOCAL_WHISPER = True   # Enable local
USE_GROQ_STT = False       # Disable Groq
```

---

## 📈 Optimization Tips

### For Speed
- Use "tiny" or "base" model
- Close background apps
- Use SSD (not HDD)

### For Accuracy
- Use "small" or "medium" model
- Speak clearly into microphone
- Reduce background noise
- Use quality microphone

### For Privacy
- Local Whisper keeps everything offline
- Audio never sent to cloud
- Perfect for sensitive datacenter work

---

## 🌐 Language Support

Local Whisper supports **99 languages**!

**English (default):**
```python
# No config needed - auto-detects English
```

**Spanish:**
```python
# Auto-detects - just speak in Spanish!
```

**Multiple languages:**
```python
# Whisper auto-detects language
# Works seamlessly across languages
```

---

## 💰 Cost Comparison

### Per 1000 queries (avg 5 seconds each)

| Method | Cost | Internet | Setup |
|--------|------|----------|-------|
| **Local Whisper** | **$0** | No | Easy |
| **Groq API** | ~$0 (free tier) | Yes | Easy |
| **OpenAI API** | ~$5 | Yes | Easy |

**Local Whisper saves $60/year if you do 10 queries/day!**

---

## ✅ Recommendation Summary

**Best overall:** Local Whisper with "base" model
- Free forever
- Good accuracy
- Fast enough
- Works offline
- Private

**If you need faster:** Local Whisper with "tiny"
- Slightly lower accuracy
- Very fast
- Still free & offline

**If local is too slow:** Groq API
- Free (for now)
- Very fast
- Requires internet

**If you want best quality:** Local Whisper with "small" or "medium"
- Better accuracy
- Still free
- Slightly slower

---

## 🚀 Quick Start Commands

```bash
# Install
pip install faster-whisper

# Configure
nano config.py
# Set: USE_LOCAL_WHISPER = True
# Set: LOCAL_WHISPER_MODEL = "base"

# Run
python biobot_voice.py

# First run downloads model (1-2 min)
# Subsequent runs are instant!
```

---

## 📚 Additional Resources

- **faster-whisper docs:** https://github.com/guillaumekln/faster-whisper
- **Model comparison:** https://github.com/openai/whisper#available-models-and-languages
- **HuggingFace models:** https://huggingface.co/Systran

---

## 🎉 Benefits Recap

Using local Whisper means:
- ✅ No API keys needed (for STT)
- ✅ No costs ever
- ✅ Works offline
- ✅ Complete privacy
- ✅ No rate limits
- ✅ No network latency
- ✅ Same quality as OpenAI
- ✅ Easy to set up

**Just one command: `pip install faster-whisper`**

Enjoy free, offline speech-to-text! 🎤
