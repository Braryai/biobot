#!/usr/bin/env python3
"""
BioBot Voice Client - Voice-controlled AI assistant for datacenter technicians
Captures audio/screenshots and queries Open WebUI with RAG-enabled knowledge base

Author: BioBot Team
Python: 3.11+
Platform: macOS
"""

import subprocess
from pathlib import Path
from datetime import datetime
from pynput import keyboard
import sounddevice as sd
import numpy as np
import wave
from openai import OpenAI
import httpx
import json
import base64
import os
from typing import Optional, Dict, Any, List
import tempfile
import importlib

# Import configuration
try:
    from config import (
        OPENAI_API_KEY,
        OPENAI_STT_MODEL,
        TRIGGER_KEY_WITH_SCREENSHOT,
        TRIGGER_KEY_AUDIO_ONLY,
        AUDIO_SAMPLE_RATE,
        AUDIO_CHANNELS,
        MAX_RECORDING_DURATION,
        OPENWEBUI_URL,
        OPENWEBUI_TOKEN,
        KNOWLEDGE_ID,
        DEFAULT_MODEL,
        TEXT_ONLY_MODEL,
        USE_LOCAL_WHISPER,
        LOCAL_WHISPER_MODEL,
        USE_GROQ_STT,
        GROQ_API_KEY,
        USE_TTS,
        TOTALGPT_API_KEY,
        TTS_VOICE,
        TTS_SPEED,
        TTS_LANG_CODE
    )
except ImportError:
    print("Error: config.py not found!")
    print("Please create config.py with your settings.")
    print("See config.py.example for reference.")
    exit(1)


# ============ PERMISSION CHECKS ============
def check_accessibility_permission() -> bool:
    """Check if app has Accessibility permissions (for keyboard monitoring)."""
    try:
        from Quartz import (
            CGPreflightScreenCaptureAccess,
            AXIsProcessTrusted
        )
        return AXIsProcessTrusted()
    except ImportError:
        # If Quartz not available, assume permission granted
        return True
    except Exception as e:
        print(f"Could not check accessibility permission: {e}")
        return True


def check_screen_recording_permission() -> bool:
    """Check if app has Screen Recording permissions."""
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        return CGPreflightScreenCaptureAccess()
    except ImportError:
        return True
    except Exception as e:
        print(f"Could not check screen recording permission: {e}")
        return True


def check_microphone_permission() -> bool:
    """Check if app has Microphone permissions."""
    try:
        # Try to open an audio stream briefly
        import sounddevice as sd
        with sd.InputStream(samplerate=16000, channels=1):
            pass
        return True
    except Exception:
        return False


def check_all_permissions() -> bool:
    """Check all required permissions and display warnings if missing."""
    all_granted = True
    
    print("\nChecking system permissions...")
    
    # Accessibility (for keyboard monitoring)
    if not check_accessibility_permission():
        all_granted = False
        print("\n⚠️  WARNING: Accessibility permission not granted!")
        print("   BioBot needs this to monitor trigger key presses.")
        print("   ")
        print("   To grant permission:")
        print("   1. Open System Settings > Privacy & Security > Accessibility")
        print("   2. Click the lock to make changes")
        print("   3. Add Terminal (or your terminal app) to the list")
        print("   4. Restart BioBot")
    else:
        print("   ✓ Accessibility: Granted")
    
    # Screen Recording
    if not check_screen_recording_permission():
        all_granted = False
        print("\n⚠️  WARNING: Screen Recording permission not granted!")
        print("   BioBot needs this to capture screenshots.")
        print("   ")
        print("   To grant permission:")
        print("   1. Open System Settings > Privacy & Security > Screen Recording")
        print("   2. Click the lock to make changes")
        print("   3. Add Terminal (or your terminal app) to the list")
        print("   4. Restart BioBot")
    else:
        print("   ✓ Screen Recording: Granted")
    
    # Microphone
    if not check_microphone_permission():
        all_granted = False
        print("\n⚠️  WARNING: Microphone permission not granted!")
        print("   BioBot needs this to record audio.")
        print("   ")
        print("   To grant permission:")
        print("   1. Open System Settings > Privacy & Security > Microphone")
        print("   2. Click the lock to make changes")
        print("   3. Add Terminal (or your terminal app) to the list")
        print("   4. Restart BioBot")
    else:
        print("   ✓ Microphone: Granted")
    
    if not all_granted:
        print("\n⚠️  Some permissions are missing. BioBot may not work correctly.")
        print("   Continue anyway? (y/N): ", end="", flush=True)
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            return False
    
    return True


# ============ AUDIO FEEDBACK ============
def play_beep(frequency: int = 800, duration: float = 0.1, volume: float = 0.3):
    """Play a simple beep tone for user feedback.
    
    Args:
        frequency: Frequency in Hz (higher = higher pitch)
        duration: Duration in seconds
        volume: Volume (0.0 to 1.0)
    """
    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = volume * np.sin(2 * np.pi * frequency * t)
        sd.play(wave, sample_rate)
        sd.wait()
    except Exception as e:
        # Silent fail - beeps are optional feedback
        pass


def play_double_beep():
    """Play double beep for screenshot mode start."""
    play_beep(frequency=800, duration=0.08, volume=0.25)
    sd.sleep(50)
    play_beep(frequency=800, duration=0.08, volume=0.25)


def play_start_beep():
    """Play single beep for audio-only mode start."""
    play_beep(frequency=600, duration=0.1, volume=0.25)


def play_stop_beep():
    """Play beep when recording stops."""
    play_beep(frequency=400, duration=0.15, volume=0.2)


def play_submit_beep():
    """Play confirmation beep when message is submitted."""
    play_beep(frequency=1000, duration=0.1, volume=0.25)
    sd.sleep(50)
    play_beep(frequency=1200, duration=0.1, volume=0.25)


# ============ AUDIO RECORDING ============
class AudioRecorder:
    """Simple push-to-talk audio recorder."""
    def __init__(self):
        self.is_recording = False
        self.audio_chunks = []
        self.stream = None
        self.capture_screenshot = True  # Track whether to capture screenshot
        self.use_region_selection = False  # Track whether to use region selector
    
    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.audio_chunks = []
        
        if self.use_region_selection:
            mode = "with region selection"
        elif self.capture_screenshot:
            mode = "with screenshot"
        else:
            mode = "audio only"
        print(f"Recording {mode}... (release key to stop)")
        print("   Audio levels: ", end="", flush=True)
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"\nStatus: {status}")
            
            # Calculate RMS (volume level)
            rms = np.sqrt(np.mean(indata**2))
            
            # Store audio data
            self.audio_chunks.append(indata.copy())
            
            # Visual feedback
            if rms > 0.02:
                print("█", end="", flush=True)
            elif rms > 0.01:
                print("▓", end="", flush=True)
            else:
                print(".", end="", flush=True)
        
        try:
            self.stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype='float32',
                blocksize=1024,
                callback=audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"\nError starting recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording and save audio file."""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            print("\nRecording stopped")
            
            if not self.audio_chunks:
                print("No audio recorded")
                return None
            
            # Combine all chunks
            audio_data = np.concatenate(self.audio_chunks, axis=0)
            
            # Save to WAV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = Path(f"/tmp/biobot_audio_{timestamp}.wav")
            
            # Convert to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(str(audio_path), 'wb') as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            
            duration = len(audio_data) / AUDIO_SAMPLE_RATE
            print(f"Audio saved: {audio_path} ({duration:.1f} seconds)")
            return audio_path
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None


# ============ TRANSCRIPTION ============
def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio using local Whisper, Groq API, or OpenAI API."""
    try:
        print("Transcribing audio...")
        
        # Option 1: Local Whisper (faster-whisper) - FREE & OFFLINE
        if USE_LOCAL_WHISPER:
            try:
                from faster_whisper import WhisperModel
                
                print(f"   Using local Whisper ({LOCAL_WHISPER_MODEL})...")
                
                # Load model (cached after first run)
                model = WhisperModel(LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8")
                
                # Transcribe
                segments, info = model.transcribe(audio_path, beam_size=5)
                transcript = " ".join([segment.text for segment in segments])
                
                print(f"Transcription: \"{transcript}\"")
                return transcript.strip()
                
            except ImportError:
                print("faster-whisper not installed, falling back to API...")
                print("   Install with: pip install faster-whisper")
                # Fall through to API options
            except Exception as e:
                print(f"Local Whisper error: {e}, falling back to API...")
                # Fall through to API options
        
        # Option 2: Groq API - FREE (for now) & FAST
        if USE_GROQ_STT and GROQ_API_KEY:
            print("   Using Groq Whisper API...")
            client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    response_format="text"
                )
            
            print(f"Transcription: \"{transcript}\"")
            return transcript.strip()
        
        # Option 3: OpenAI API - PAID but RELIABLE
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your-"):
            print("   Using OpenAI Whisper API...")
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=OPENAI_STT_MODEL,
                    file=audio_file,
                    response_format="text"
                )
            
            print(f"Transcription: \"{transcript}\"")
            return transcript.strip()
        
        # No valid STT option configured
        print("No STT service configured!")
        print("   Enable one of:")
        print("   - USE_LOCAL_WHISPER = True (free, offline)")
        print("   - USE_GROQ_STT = True with GROQ_API_KEY")
        print("   - Set OPENAI_API_KEY")
        return None
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


# ============ SCREENSHOT CAPTURE ============
def capture_screenshot(use_region_selection: bool = False) -> Optional[str]:
    """Capture focused window, region selection, or fall back to full screen.
    
    Args:
        use_region_selection: If True, show region selector overlay
        
    Returns:
        Path to screenshot file or None if failed
    """
    if use_region_selection:
        try:
            from region_selector import RegionSelector, capture_region_screenshot
            
            print("   Opening region selector...")
            selector = RegionSelector()
            region = selector.select_region()
            
            if region:
                x, y, w, h = region
                print(f"   Selected region: {w}x{h} at ({x}, {y})")
            else:
                print("   No region selected, using focused window")
            
            return capture_region_screenshot(region)
            
        except ImportError as e:
            print(f"   Region selector not available: {e}")
            print("   Falling back to normal screenshot...")
        except Exception as e:
            print(f"   Region selection failed: {e}")
            print("   Falling back to normal screenshot...")
    
    # Normal screenshot capture (focused window or full screen)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = Path(f"/tmp/biobot_screenshot_{timestamp}.png")
    
    try:
        print("   Attempting to capture focused window...")
        
        window_captured = False
        
        try:
            # Import Quartz for window capture
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
            )
            from Cocoa import NSWorkspace
            
            # Get frontmost app
            frontmost_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            app_pid = frontmost_app.processIdentifier()
            app_name = frontmost_app.localizedName()
            print(f"   Frontmost app: {app_name} (PID: {app_pid})")
            
            # Get window list
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID
            )
            
            # Find frontmost window
            target_window_id = None
            for window in window_list:
                if window.get('kCGWindowOwnerPID') == app_pid:
                    layer = window.get('kCGWindowLayer', -1)
                    bounds = window.get('kCGWindowBounds', {})
                    width = bounds.get('Width', 0)
                    height = bounds.get('Height', 0)
                    
                    # Look for a normal window (layer 0) with reasonable size
                    if layer == 0 and width > 100 and height > 100:
                        target_window_id = window.get('kCGWindowNumber')
                        print(f"   Window ID: {target_window_id}")
                        print(f"   Window bounds: {width:.0f}x{height:.0f}")
                        break
            
            if target_window_id:
                # Capture using screencapture with window ID
                result = subprocess.run(
                    ["screencapture", "-x", "-o", "-l", str(target_window_id), "-t", "png", str(screenshot_path)],
                    capture_output=True,
                    timeout=5
                )
                
                if result.returncode == 0 and screenshot_path.exists() and screenshot_path.stat().st_size > 1000:
                    print(f"Captured focused window!")
                    window_captured = True
                else:
                    print(f"   screencapture failed for window ID {target_window_id}")
            else:
                print(f"   Could not find valid window for {app_name}")
                    
        except ImportError:
            print(f"   PyObjC not available, trying simple method...")
        except Exception as e:
            print(f"   Window capture error: {e}")
        
        # If window capture didn't work, fall back to full screen
        if not window_captured:
            print(f"   Falling back to full screen capture...")
            
            result = subprocess.run(
                ["screencapture", "-x", "-t", "png", str(screenshot_path)],
                capture_output=True,
                timeout=5
            )
        
        if result.returncode != 0:
            print(f"screencapture returned code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.decode()}")
            return None
        
        # Verify file exists and has content
        if not screenshot_path.exists():
            print("✗ Screenshot file was not created")
            return None
        
        file_size = screenshot_path.stat().st_size
        if file_size < 1000:
            print(f"✗ Screenshot file too small: {file_size} bytes")
            screenshot_path.unlink()
            return None
        
        # Get image dimensions
        try:
            sips_result = subprocess.run(
                ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(screenshot_path)],
                capture_output=True,
                text=True,
                timeout=3
            )
            if sips_result.returncode == 0:
                lines = sips_result.stdout.split('\n')
                width = height = None
                for line in lines:
                    if 'pixelWidth:' in line:
                        width = line.split(':')[1].strip()
                    elif 'pixelHeight:' in line:
                        height = line.split(':')[1].strip()
                
                if width and height:
                    print(f"  Dimensions: {width} x {height} pixels")
        except:
            pass
        
        size_mb = file_size / (1024 * 1024)
        print(f"Captured screenshot: {screenshot_path}")
        print(f"  File size: {size_mb:.2f} MB ({file_size:,} bytes)")
        
        return str(screenshot_path)
            
    except subprocess.TimeoutExpired:
        print("✗ Screenshot capture timed out")
        return None
    except Exception as e:
        print(f"✗ Screenshot capture failed: {e}")
        return None


# ============ OPEN WEBUI API CLIENT ============
def get_available_models() -> list:
    """Get list of available models from Open WebUI."""
    try:
        headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{OPENWEBUI_URL}/api/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            
            models = [model.get('id') for model in data.get('data', [])]
            return models
    except Exception as e:
        print(f"Could not fetch models: {e}")
        return []


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_chat_in_openwebui(model: str) -> Optional[str]:
    """Create a new chat in Open WebUI and return the chat ID."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "chat": {
                "name": f"BioBot - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "models": [model]
            }
        }
        
        url = f"{OPENWEBUI_URL}/api/v1/chats/new"
        
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            if 'id' in result:
                chat_id = result['id']
                print(f"💬 Created new chat: {chat_id}")
                return chat_id
            else:
                print(f"Chat created but no ID returned: {result}")
                return None
                
    except Exception as e:
        print(f"Could not create chat in UI: {e}")
        return None


def upload_image_to_openwebui(screenshot_path: str) -> Optional[Dict[str, Any]]:
    """Upload image using official /api/v1/files/ endpoint (multipart form)."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}"
        }
        
        # Read image file
        filename = Path(screenshot_path).name
        
        with open(screenshot_path, 'rb') as f:
            files = {'file': (filename, f, 'image/png')}
            
            url = f"{OPENWEBUI_URL}/api/v1/files/"
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, files=files)
                response.raise_for_status()
                result = response.json()
                
                print(f"Image uploaded: {result.get('id')}")
                return result
                
    except Exception as e:
        print(f"Could not upload image: {e}")
        return None


def add_message_to_chat(chat_id: str, user_message: str, assistant_message: str, file_data: Optional[Dict[str, Any]] = None) -> bool:
    """Add messages to chat WITHOUT overwriting history (like test_multiple_images.py)."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Get current chat
        get_url = f"{OPENWEBUI_URL}/api/v1/chats/{chat_id}"
        
        with httpx.Client(timeout=30.0) as client:
            # Get existing messages
            get_response = client.get(get_url, headers=headers)
            get_response.raise_for_status()
            chat_data = get_response.json()
            
            existing_messages = chat_data["chat"].get("messages", [])
            
            # Build user message
            user_msg = {
                "role": "user",
                "content": user_message
            }
            
            # Add attachment if we have image
            if file_data:
                attachment = {
                    "type": "image",
                    "file_id": file_data["id"],
                    "url": f"/api/v1/files/{file_data['id']}/content",
                    "filename": file_data.get("filename", "screenshot.png"),
                    "content_type": file_data.get("meta", {}).get("content_type", "image/png"),
                    "size": file_data.get("meta", {}).get("size", 0)
                }
                user_msg["attachments"] = [attachment]
            
            # Build assistant message
            assistant_msg = {
                "role": "assistant",
                "content": assistant_message
            }
            
            # APPEND new messages (don't overwrite!)
            all_messages = existing_messages + [user_msg, assistant_msg]
            
            # Save
            chat_data["chat"]["messages"] = all_messages
            update_url = f"{OPENWEBUI_URL}/api/v1/chats/{chat_id}"
            update_response = client.post(update_url, json={"chat": chat_data["chat"]}, headers=headers)
            update_response.raise_for_status()
            
            print(f"Message saved to chat")
            return True
                
    except Exception as e:
        print(f"Could not save to chat: {e}")
        return False


def query_openwebui(query_text: str, screenshot_path: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None, chat_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Send query to /api/chat/completions with inline base64 (for response only, not persistence)."""
    try:
        print("Querying model...")
        
        # Choose model based on whether we have a screenshot
        model_to_use = DEFAULT_MODEL if screenshot_path else TEXT_ONLY_MODEL
        
        # Build messages array with conversation history
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
            print(f"   📜 Including {len(conversation_history)} previous messages")
        
        # Add current user message with inline base64 if image
        if screenshot_path:
            image_base64 = encode_image_to_base64(screenshot_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": query_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": query_text
            })
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "stream": False
        }
        
        # Send request (only to get response, NOT for persistence)
        url = f"{OPENWEBUI_URL}/api/chat/completions"
        print(f"   Using model: {model_to_use}")
        
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        
        result = response.json()
        print("Model response received")
        
        # Extract response content
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
        else:
            content = ""
        
        # Return simplified result
        return {
            "content": content,
            "_query_text": query_text,
            "_has_screenshot": screenshot_path is not None
        }
        
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        print(f"Request Error: {e}")
        print("   Is Open WebUI running? Check the URL in config.py")
        return None
    except Exception as e:
        print(f"Error querying Open WebUI: {e}")
        return None


def text_to_speech(text: str) -> Optional[str]:
    """Convert text to speech using TotalGPT API and play it."""
    try:
        if not USE_TTS or not TOTALGPT_API_KEY:
            return None
            
        print("🔊 Converting response to speech...")
        
        headers = {
            "Authorization": f"Bearer {TOTALGPT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "voice": TTS_VOICE,
            "model": "TTS-hexgrad-Kokoro-82M",
            "response_format": "mp3",
            "speed": TTS_SPEED,
            "lang_code": TTS_LANG_CODE
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.totalgpt.ai/v1/audio/speech",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
        
        # Save audio to temp file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = Path(f"/tmp/biobot_tts_{timestamp}.mp3")
        
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        print("Speech generated, playing...")
        
        # Play audio using afplay (macOS built-in)
        subprocess.run(["afplay", str(audio_path)], check=True)
        
        # Clean up
        audio_path.unlink()
        print("Speech playback complete")
        
        return str(audio_path)
        
    except httpx.HTTPStatusError as e:
        print(f"TTS API Error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def display_response(response: Dict[str, Any]):
    """Display the response from model."""
    try:
        # Extract content (simplified structure)
        content = response.get('content', '')
        
        if content:
            print("\n" + "="*60)
            print("BIOBOT RESPONSE:")
            print("="*60)
            print(content)
            print("="*60)
            
            # Text-to-speech if enabled
            if USE_TTS:
                text_to_speech(content)
        else:
            print("No content in response")
            
    except Exception as e:
        print(f"Error displaying response: {e}")


# ============ MAIN PROCESSING FUNCTION ============
# Global variables to store conversation state
CONVERSATION_HISTORY = []  # Text-only for API calls to avoid huge payloads
CURRENT_CHAT_ID = None  # Open WebUI chat ID for frontend display
MODIFIER_KEYS_PRESSED = set()  # Track which modifier keys are currently pressed

def process_query(audio_path: str, capture_screenshot_flag: bool, use_region_selection: bool = False):
    """Process the complete workflow: transcribe, optionally screenshot, query, display.
    
    Args:
        audio_path: Path to recorded audio file
        capture_screenshot_flag: Whether to capture screenshot
        use_region_selection: Whether to use region selector for screenshot
    """
    global CONVERSATION_HISTORY, CURRENT_CHAT_ID
    screenshot_path = None
    
    try:
        print("\n" + "="*60)
        print("PROCESSING QUERY...")
        print("="*60)
        
        if not audio_path:
            print("No audio recorded, aborting...")
            return
        
        # Step 1: Transcribe audio
        message_text = transcribe_audio(audio_path)
        if not message_text:
            print("Transcription failed, aborting...")
            return
        
        # Step 2: Capture screenshot if requested
        if capture_screenshot_flag:
            print("Capturing screenshot...")
            screenshot_path = capture_screenshot(use_region_selection)
            if not screenshot_path:
                print("Screenshot capture failed, continuing without image...")
        else:
            print("Skipping screenshot (audio-only mode)")
        
        # Step 3: Create chat on first message
        if CURRENT_CHAT_ID is None:
            CURRENT_CHAT_ID = create_chat_in_openwebui("Llama-3.2-11B-Vision-Instruct")
            if not CURRENT_CHAT_ID:
                print("Could not create chat")
                return
        
        # Step 4: Upload image if we have one
        file_data = None
        if screenshot_path:
            file_data = upload_image_to_openwebui(screenshot_path)
            if not file_data:
                print("Image upload failed, continuing without attachment...")
        
        # Step 5: Query model (inline base64 for quick response)
        response = query_openwebui(message_text, screenshot_path, CONVERSATION_HISTORY, None)
        
        if not response:
            print("No response from model")
            return
        
        # Extract assistant response
        assistant_message = response.get('content', '')
        
        if assistant_message:
            # Add to conversation history (text-only for efficiency)
            query_text_only = response.get('_query_text', message_text)
            
            CONVERSATION_HISTORY.append({
                "role": "user",
                "content": query_text_only
            })
            CONVERSATION_HISTORY.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Save to chat with attachments (like test_multiple_images.py)
            add_message_to_chat(CURRENT_CHAT_ID, message_text, assistant_message, file_data)
            play_submit_beep()  # Audio confirmation that message was submitted
        
        # Step 6: Display the response
        display_response(response)
        
        print("\n✅ Query complete!")
        print("="*60)
        print(f"Ready for next query (Right Cmd = w/ screenshot, Right Shift = audio only)...\n")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Try again (Right Cmd = w/ screenshot, Right Shift = audio only)...\n")
    
    finally:
        # Save screenshot to permanent folder for debugging
        if screenshot_path and Path(screenshot_path).exists():
            try:
                # Create screenshots folder
                screenshots_dir = Path(__file__).parent / "screenshots"
                screenshots_dir.mkdir(exist_ok=True)
                
                # Copy screenshot to permanent location
                import shutil
                permanent_path = screenshots_dir / Path(screenshot_path).name
                shutil.copy2(screenshot_path, permanent_path)
                print(f"Screenshot saved: {permanent_path}")
            except Exception as e:
                print(f"Could not save screenshot: {e}")
        
        # Clean up temporary files
        if audio_path and Path(audio_path).exists():
            try:
                Path(audio_path).unlink()
                print(f"Cleaned up audio file")
            except Exception as e:
                print(f"Could not delete audio file: {e}")
        
        if screenshot_path and Path(screenshot_path).exists():
            try:
                Path(screenshot_path).unlink()
                print(f"Cleaned up temp screenshot")
            except Exception as e:
                print(f"Could not delete screenshot file: {e}")


# ============ KEYBOARD LISTENER ============
def get_trigger_key_map(trigger_key_screenshot=None, trigger_key_audio=None):
    """Map trigger key strings to pynput Key objects.
    
    Args:
        trigger_key_screenshot: Override for screenshot trigger
        trigger_key_audio: Override for audio-only trigger
    """
    from pynput.keyboard import Key
    
    # Use provided keys or fall back to config
    key1 = trigger_key_screenshot or TRIGGER_KEY_WITH_SCREENSHOT
    key2 = trigger_key_audio or TRIGGER_KEY_AUDIO_ONLY
    
    return {
        'cmd_r': Key.cmd_r,
        'cmd': Key.cmd,
        'shift_r': Key.shift_r,
        'shift': Key.shift,
        'alt_r': Key.alt_r,
        'alt': Key.alt,
        'ctrl_r': Key.ctrl_r,
        'ctrl': Key.ctrl,
        'key1': key1,  # Store configured keys
        'key2': key2,
    }


def check_key_match(key, trigger_key_str, trigger_map=None):
    """Check if a key matches a trigger key string.
    
    Args:
        key: The pynput key object
        trigger_key_str: String identifier for the trigger key
        trigger_map: Optional pre-built trigger map to use
    """
    try:
        if trigger_map is None:
            trigger_map = get_trigger_key_map()
        
        # Check for mapped modifier keys
        if trigger_key_str.lower() in trigger_map:
            return key == trigger_map[trigger_key_str.lower()]
        
        # Check for function keys (f1-f12, etc.)
        if hasattr(key, 'name'):
            return key.name.lower() == trigger_key_str.lower()
        
        # Fallback for character keys
        if hasattr(key, 'char'):
            return key.char == trigger_key_str
    except:
        pass
    return False


def on_press(key, recorder, trigger_key_screenshot=None, trigger_key_audio=None):
    """Handle key press events - start recording.
    
    Args:
        key: The pynput key object
        recorder: AudioRecorder instance
        trigger_key_screenshot: Key for screenshot+audio mode
        trigger_key_audio: Key for audio-only mode
    """
    global MODIFIER_KEYS_PRESSED
    
    # Use config defaults if not provided
    key1 = trigger_key_screenshot or TRIGGER_KEY_WITH_SCREENSHOT
    key2 = trigger_key_audio or TRIGGER_KEY_AUDIO_ONLY
    
    # Build trigger map once
    trigger_map = get_trigger_key_map(key1, key2)
    
    try:
        # Check for Escape key to reset conversation
        from pynput.keyboard import Key
        
        # Track modifier keys
        if key in (Key.cmd, Key.cmd_r, Key.cmd_l):
            MODIFIER_KEYS_PRESSED.add('cmd')
        elif key in (Key.shift, Key.shift_r, Key.shift_l):
            MODIFIER_KEYS_PRESSED.add('shift')
        elif key in (Key.alt, Key.alt_r, Key.alt_l):
            MODIFIER_KEYS_PRESSED.add('alt')
        elif key in (Key.ctrl, Key.ctrl_r, Key.ctrl_l):
            MODIFIER_KEYS_PRESSED.add('ctrl')
        
        if key == Key.esc:
            if CONVERSATION_HISTORY or CURRENT_CHAT_ID:
                print("\n" + "="*60)
                print("CONVERSATION RESET - Starting fresh...")
                print("="*60 + "\n")
                CONVERSATION_HISTORY = []
                CURRENT_CHAT_ID = None
            return
        
        # Check for screenshot + audio trigger
        if check_key_match(key, key1, trigger_map):
            if not recorder.is_recording:
                # Check if Cmd+Shift for region selection
                if 'cmd' in MODIFIER_KEYS_PRESSED and 'shift' in MODIFIER_KEYS_PRESSED:
                    print("\n" + "="*60)
                    print(f"REGION SELECTION MODE - Cmd+Shift+{key1.replace('_r', ' (Right)').replace('_', ' ').title()}")
                    print("="*60)
                    recorder.use_region_selection = True
                    recorder.capture_screenshot = True
                else:
                    play_double_beep()  # Audio feedback for screenshot mode
                    recorder.use_region_selection = False
                    recorder.capture_screenshot = True
                    key_display = key1.replace('_r', ' (Right)').replace('_', ' ').title()
                    print("\n" + "="*60)
                    print(f"{key_display} PRESSED - Recording with screenshot...")
                    print("="*60)
                recorder.start_recording()
        
        # Check for audio-only trigger
        elif check_key_match(key, key2, trigger_map):
            if not recorder.is_recording:
                play_start_beep()  # Audio feedback for audio-only mode
                recorder.use_region_selection = False
                recorder.capture_screenshot = False
                key_display = key2.replace('_r', ' (Right)').replace('_', ' ').title()
                print("\n" + "="*60)
                print(f"{key_display} PRESSED - Recording audio only...")
                print("="*60)
                recorder.start_recording()
    except Exception as e:
        print(f"Error in key press handler: {e}")


def on_release(key, recorder, trigger_key_screenshot=None, trigger_key_audio=None):
    """Handle key release events - stop recording and process.
    
    Args:
        key: The pynput key object
        recorder: AudioRecorder instance
        trigger_key_screenshot: Key for screenshot+audio mode
        trigger_key_audio: Key for audio-only mode
    """
    global MODIFIER_KEYS_PRESSED
    
    # Use config defaults if not provided
    key1 = trigger_key_screenshot or TRIGGER_KEY_WITH_SCREENSHOT
    key2 = trigger_key_audio or TRIGGER_KEY_AUDIO_ONLY
    
    # Build trigger map once
    trigger_map = get_trigger_key_map(key1, key2)
    
    try:
        # Track modifier key releases
        from pynput.keyboard import Key
        if key in (Key.cmd, Key.cmd_r, Key.cmd_l):
            MODIFIER_KEYS_PRESSED.discard('cmd')
        elif key in (Key.shift, Key.shift_r, Key.shift_l):
            MODIFIER_KEYS_PRESSED.discard('shift')
        elif key in (Key.alt, Key.alt_r, Key.alt_l):
            MODIFIER_KEYS_PRESSED.discard('alt')
        elif key in (Key.ctrl, Key.ctrl_r, Key.ctrl_l):
            MODIFIER_KEYS_PRESSED.discard('ctrl')
        
        # Check if either trigger key was released
        if check_key_match(key, key1, trigger_map) or check_key_match(key, key2, trigger_map):
            if recorder.is_recording:
                play_stop_beep()  # Audio feedback when recording stops
                capture_screenshot_flag = recorder.capture_screenshot
                use_region_flag = recorder.use_region_selection
                audio_path = recorder.stop_recording()
                if audio_path:
                    process_query(audio_path, capture_screenshot_flag, use_region_flag)
    except Exception as e:
        print(f"Error in key release handler: {e}")


# ============ INTERACTIVE KEY SETUP ============
def setup_trigger_keys():
    """Interactive wizard to configure trigger keys on first run."""
    print("\n" + "="*60)
    print("FIRST-TIME SETUP: Configure Your Trigger Keys")
    print("="*60)
    print("\nBioBot needs two trigger keys for different modes:")
    print("\n  Mode 1: Audio + Screenshot")
    print("     Takes a screenshot and records audio simultaneously")
    print("\n  Mode 2: Audio Only")
    print("     Records audio without taking a screenshot")
    print("\n" + "-"*60)
    
    key_mapping = {
        'Key.cmd_r': 'cmd_r',
        'Key.shift_r': 'shift_r',
        'Key.alt_r': 'alt_r',
        'Key.ctrl_r': 'ctrl_r',
        'Key.cmd': 'cmd',
        'Key.shift': 'shift',
        'Key.alt': 'alt',
        'Key.ctrl': 'ctrl',
    }
    
    def capture_key():
        """Capture a single key press with hold confirmation."""
        import time
        import sys
        
        captured_key = [None]
        key_pressed = [False]
        key_released = [False]
        
        def on_press(key):
            key_str = str(key)
            if key_str in key_mapping and not key_pressed[0]:
                captured_key[0] = key_mapping[key_str]
                key_pressed[0] = True
        
        def on_release(key):
            if key_pressed[0] and not key_released[0]:
                key_released[0] = True
                return False  # Stop listener
        
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        # Wait for initial key press
        while not key_pressed[0]:
            time.sleep(0.01)
        
        # Show progress bar while key is held
        print("\nHold the key for 2 seconds...")
        progress_duration = 2.0  # seconds
        bar_width = 40
        start_time = time.time()
        
        while time.time() - start_time < progress_duration:
            if key_released[0]:
                # Key released too early
                listener.stop()
                print("\n\nKey released too early! Please hold for 2 seconds.")
                return None
            
            elapsed = time.time() - start_time
            progress = min(elapsed / progress_duration, 1.0)
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            percent = int(progress * 100)
            
            sys.stdout.write(f'\r[{bar}] {percent}%')
            sys.stdout.flush()
            time.sleep(0.05)
        
        print("\n✓ Key captured!")
        listener.stop()
        
        return captured_key[0]
    
    # Capture first key
    print("\nSTEP 1/2: Press and HOLD the key for 'Audio + Screenshot' mode")
    print("          Recommended: Right Command (⌘)")
    print("\nPress and hold your key now...")
    key1 = capture_key()
    
    while not key1:
        print("\nTry again - Press and HOLD for 2 seconds...")
        key1 = capture_key()
    
    key1_display = key1.replace('_r', ' (Right)').replace('_', ' ').title()
    print(f"Registered: {key1_display}")
    
    # Capture second key
    print("\nSTEP 2/2: Press and HOLD the key for 'Audio Only' mode")
    print("          Recommended: Right Shift (⇧)")
    print("\nPress and hold your key now...")
    key2 = capture_key()
    
    while not key2:
        print("\nTry again - Press and HOLD for 2 seconds...")
        key2 = capture_key()
    
    while key2 == key1:
        print("\nERROR: You must choose a different key from the first one.")
        print("Try again - Press and HOLD for 2 seconds...")
        key2 = capture_key()
    
    key2_display = key2.replace('_r', ' (Right)').replace('_', ' ').title()
    print(f"Registered: {key2_display}")
    
    # Confirm selection
    print("\n" + "="*60)
    print("KEY CONFIGURATION SUMMARY")
    print("="*60)
    print(f"\n  Audio + Screenshot: {key1_display}")
    print(f"  Audio Only:         {key2_display}")
    print(f"\n  Conversation Reset: ESC (always)")
    print("\n" + "="*60)
    print("\nConfirm this configuration? (Y/n): ", end="", flush=True)
    
    confirmation = input().strip().lower()
    if confirmation and confirmation not in ['y', 'yes']:
        print("\nSetup cancelled. Please run again to reconfigure.")
        return None
    
    return key1, key2


def update_config_file(key1, key2):
    """Update config.py with the selected trigger keys."""
    config_path = Path(__file__).parent / "config.py"
    
    if not config_path.exists():
        print("\nERROR: config.py not found!")
        print("Please create config.py from config.py.example first.")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Update the trigger keys
    import re
    
    # Replace TRIGGER_KEY_WITH_SCREENSHOT
    config_content = re.sub(
        r"TRIGGER_KEY_WITH_SCREENSHOT\s*=\s*['\"].*?['\"]",
        f"TRIGGER_KEY_WITH_SCREENSHOT = '{key1}'",
        config_content
    )
    
    # Replace TRIGGER_KEY_AUDIO_ONLY
    config_content = re.sub(
        r"TRIGGER_KEY_AUDIO_ONLY\s*=\s*['\"].*?['\"]",
        f"TRIGGER_KEY_AUDIO_ONLY = '{key2}'",
        config_content
    )
    
    # Write back to config
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print("\nConfiguration saved successfully!")
    print(f"Updated: {config_path}")
    
    # Reload the config module to get updated values
    import config
    importlib.reload(config)
    
    return config.TRIGGER_KEY_WITH_SCREENSHOT, config.TRIGGER_KEY_AUDIO_ONLY


def check_first_run():
    """Check if this is the first run and trigger keys need configuration.
    
    Returns:
        tuple or None: (key1, key2) if custom keys were configured, None to use defaults
    """
    # Check if trigger keys are still at default values
    if TRIGGER_KEY_WITH_SCREENSHOT == 'cmd_r' and TRIGGER_KEY_AUDIO_ONLY == 'shift_r':
        print("\nDetected default key configuration.")
        print("\nWould you like to configure custom trigger keys? (y/N): ", end="", flush=True)
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            keys = setup_trigger_keys()
            if keys:
                key1, key2 = keys
                result = update_config_file(key1, key2)
                if result:
                    print("\nConfiguration updated! Continuing with new keys...")
                    return result  # Return the new keys
        else:
            print("\nUsing default keys. You can reconfigure later by editing config.py")
    
    return None  # Use current config


# ============ MAIN ENTRY POINT ============
def main():
    """Main entry point for BioBot Voice Client."""
    print("="*60)
    print("BioBot Voice Client - Datacenter AI Assistant")
    print("="*60)
    
    # Check system permissions first
    if not check_all_permissions():
        print("\nExiting due to missing permissions.")
        return
    
    # Check for first-time setup
    custom_keys = check_first_run()
    
    # Use custom keys if configured, otherwise use imported values
    if custom_keys:
        trigger_key_screenshot, trigger_key_audio = custom_keys
    else:
        trigger_key_screenshot = TRIGGER_KEY_WITH_SCREENSHOT
        trigger_key_audio = TRIGGER_KEY_AUDIO_ONLY
    
    # Fetch available models
    print("\nFetching available models...")
    available_models = get_available_models()
    if available_models:
        print(f"Found {len(available_models)} models")
        if DEFAULT_MODEL in available_models:
            print(f"Configured model '{DEFAULT_MODEL}' is available")
        else:
            print(f"WARNING: Configured model '{DEFAULT_MODEL}' not found!")
            print(f"   Available models: {', '.join(available_models[:5])}")
            if len(available_models) > 5:
                print(f"   ... and {len(available_models) - 5} more")
    
    # Check if at least one STT option is configured
    stt_configured = False
    
    if USE_LOCAL_WHISPER:
        print("   Using local Whisper (offline, free)")
        stt_configured = True
    elif USE_GROQ_STT and GROQ_API_KEY and not GROQ_API_KEY.startswith("your-"):
        print("   Using Groq API for STT")
        stt_configured = True
    elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your-"):
        print("   Using OpenAI API for STT")
        stt_configured = True
    
    if not stt_configured:
        print("\nERROR: No STT (Speech-to-Text) service configured!")
        print("\nYou have 3 options:")
        print("\n1. LOCAL WHISPER (Recommended - Free & Offline)")
        print("   Set: USE_LOCAL_WHISPER = True")
        print("   Install: pip install faster-whisper")
        print("\n2. GROQ API (Free for now)")
        print("   Set: USE_GROQ_STT = True")
        print("   Get key from: https://console.groq.com/keys")
        print("\n3. OPENAI API (Paid but reliable)")
        print("   Get key from: https://platform.openai.com/api-keys")
        exit(1)
    
    if not OPENWEBUI_TOKEN or OPENWEBUI_TOKEN.startswith("your-"):
        print("\nERROR: Open WebUI token not set!")
        print("Please edit config.py and set your Open WebUI API token.")
        exit(1)
    
    # Test Open WebUI connection
    print("\nTesting connection to Open WebUI...")
    print(f"   URL: {OPENWEBUI_URL}")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
            response = client.get(f"{OPENWEBUI_URL}/api/config", headers=headers)
            response.raise_for_status()
            print("Connected to Open WebUI successfully!")
    except Exception as e:
        print(f"Could not connect to Open WebUI: {e}")
        print("   Please check that:")
        print("   1. Open WebUI is running")
        print("   2. The URL in config.py is correct")
        print("   3. Your API token is valid")
        exit(1)
    
    # Create audio recorder
    recorder = AudioRecorder()
    
    # Format the key names nicely for display
    key1_display = trigger_key_screenshot.replace('_r', ' (Right)').replace('_', ' ').title()
    key2_display = trigger_key_audio.replace('_r', ' (Right)').replace('_', ' ').title()
    
    # Get trigger key map with custom keys
    trigger_map = get_trigger_key_map(trigger_key_screenshot, trigger_key_audio)
    
    print("\n" + "="*60)
    print("READY! Three modes:")
    print("")
    print(f"   {key1_display} - Audio + Screenshot")
    print("      Hold, speak, release → sends with focused window image")
    print("")
    print(f"   Cmd+Shift+{key1_display} - Region Selection")
    print("      Opens overlay to select specific screen area")
    print("")
    print(f"   {key2_display} - Audio Only")
    print("      Hold, speak, release → sends without image")
    print("")
    print("    ESC - Reset conversation (start fresh)")
    print("")
    print("   Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    # Set up keyboard listener with custom keys
    try:
        with keyboard.Listener(
            on_press=lambda key: on_press(key, recorder, trigger_key_screenshot, trigger_key_audio),
            on_release=lambda key: on_release(key, recorder, trigger_key_screenshot, trigger_key_audio)
        ) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\n\nShutting down BioBot...")
        print("Goodbye! \n")


if __name__ == "__main__":
    main()
