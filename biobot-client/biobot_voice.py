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


# ============ AUDIO RECORDING ============
class AudioRecorder:
    """Simple push-to-talk audio recorder."""
    def __init__(self):
        self.is_recording = False
        self.audio_chunks = []
        self.stream = None
        self.capture_screenshot = True  # Track whether to capture screenshot
    
    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.audio_chunks = []
        
        mode = "with screenshot" if self.capture_screenshot else "audio only"
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
def capture_screenshot() -> Optional[str]:
    """Capture focused window or fall back to full screen."""
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

def process_query(audio_path: str, capture_screenshot_flag: bool):
    """Process the complete workflow: transcribe, optionally screenshot, query, display."""
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
            screenshot_path = capture_screenshot()
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
def get_trigger_key_map():
    """Map trigger key strings to pynput Key objects."""
    from pynput.keyboard import Key
    
    return {
        'cmd_r': Key.cmd_r,
        'cmd': Key.cmd,
        'shift_r': Key.shift_r,
        'shift': Key.shift,
        'alt_r': Key.alt_r,
        'alt': Key.alt,
        'ctrl_r': Key.ctrl_r,
        'ctrl': Key.ctrl,
    }


def check_key_match(key, trigger_key_str):
    """Check if a key matches a trigger key string."""
    try:
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


def on_press(key, recorder):
    """Handle key press events - start recording."""
    try:
        # Check for Escape key to reset conversation
        from pynput.keyboard import Key
        if key == Key.esc:
            if CONVERSATION_HISTORY or CURRENT_CHAT_ID:
                print("\n" + "="*60)
                print("CONVERSATION RESET - Starting fresh...")
                print("="*60 + "\n")
                CONVERSATION_HISTORY = []
                CURRENT_CHAT_ID = None
            return
        
        # Check for screenshot + audio trigger
        if check_key_match(key, TRIGGER_KEY_WITH_SCREENSHOT):
            if not recorder.is_recording:
                recorder.capture_screenshot = True
                key_display = TRIGGER_KEY_WITH_SCREENSHOT.replace('_r', ' (Right)').replace('_', ' ').title()
                print("\n" + "="*60)
                print(f"{key_display} PRESSED - Recording with screenshot...")
                print("="*60)
                recorder.start_recording()
        
        # Check for audio-only trigger
        elif check_key_match(key, TRIGGER_KEY_AUDIO_ONLY):
            if not recorder.is_recording:
                recorder.capture_screenshot = False
                key_display = TRIGGER_KEY_AUDIO_ONLY.replace('_r', ' (Right)').replace('_', ' ').title()
                print("\n" + "="*60)
                print(f"{key_display} PRESSED - Recording audio only...")
                print("="*60)
                recorder.start_recording()
    except Exception as e:
        print(f"Error in key press handler: {e}")


def on_release(key, recorder):
    """Handle key release events - stop recording and process."""
    try:
        # Check if either trigger key was released
        if check_key_match(key, TRIGGER_KEY_WITH_SCREENSHOT) or check_key_match(key, TRIGGER_KEY_AUDIO_ONLY):
            if recorder.is_recording:
                capture_screenshot_flag = recorder.capture_screenshot
                audio_path = recorder.stop_recording()
                if audio_path:
                    process_query(audio_path, capture_screenshot_flag)
    except Exception as e:
        print(f"Error in key release handler: {e}")


# ============ MAIN ENTRY POINT ============
def main():
    """Main entry point for BioBot Voice Client."""
    print("="*60)
    print("BioBot Voice Client - Datacenter AI Assistant")
    print("="*60)
    
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
    key1_display = TRIGGER_KEY_WITH_SCREENSHOT.replace('_r', ' (Right)').replace('_', ' ').title()
    key2_display = TRIGGER_KEY_AUDIO_ONLY.replace('_r', ' (Right)').replace('_', ' ').title()
    
    print("\n" + "="*60)
    print("✅ READY! Two modes:")
    print("")
    print(f"   {key1_display} - Audio + Screenshot")
    print("      Hold, speak, release → sends with image")
    print("")
    print(f"   {key2_display} - Audio Only")
    print("      Hold, speak, release → sends without image")
    print("")
    print("    ESC - Reset conversation (start fresh)")
    print("")
    print("   Using modifier keys prevents beeping!")
    print("   Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    # Set up keyboard listener
    try:
        with keyboard.Listener(
            on_press=lambda key: on_press(key, recorder),
            on_release=lambda key: on_release(key, recorder)
        ) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\n\nShutting down BioBot...")
        print("Goodbye! \n")


if __name__ == "__main__":
    main()
