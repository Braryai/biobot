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
from dashboard import get_dashboard, BioBotDashboard
from rich.live import Live
import threading
import time

DASHBOARD = None
LIVE_DISPLAY = None

# Import configuration
try:
    from config import (
        USE_OPENWEBUI,
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
    
    # Import standalone mode configs
    try:
        from config import STANDALONE_MODEL, STANDALONE_VISION_MODEL
    except ImportError:
        # Fallback if not defined in config
        STANDALONE_MODEL = "Qwen-Qwen3-30B-A3B"
        STANDALONE_VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
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
        self.audio_data = None
        self.recording_thread = None
        self.capture_screenshot = True
        self.use_region_selection = False
    
    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.audio_data = None
        
        if self.use_region_selection:
            mode = "with region selection"
        elif self.capture_screenshot:
            mode = "with screenshot"
        else:
            mode = "audio only"
        print(f"Recording {mode}... (release key to stop)")
        
        if DASHBOARD:
            DASHBOARD.set_processing_step("Recording audio", 5)
        
        # Start recording in background thread
        def record_thread():
            try:
                # Record with visual feedback
                print("   Audio levels: ", end="", flush=True)
                
                # Start recording (non-blocking)
                self.audio_data = sd.rec(
                    int(MAX_RECORDING_DURATION * AUDIO_SAMPLE_RATE),
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    dtype='float32'
                )
                
                # Visual feedback while recording
                start_time = time.time()
                while self.is_recording and (time.time() - start_time) < MAX_RECORDING_DURATION:
                    # Calculate current RMS
                    if self.audio_data is not None:
                        current_frame = int((time.time() - start_time) * AUDIO_SAMPLE_RATE)
                        if current_frame > 0 and current_frame < len(self.audio_data):
                            # Get last 0.1 seconds of audio
                            window_size = int(0.1 * AUDIO_SAMPLE_RATE)
                            start_idx = max(0, current_frame - window_size)
                            audio_window = self.audio_data[start_idx:current_frame]
                            
                            if len(audio_window) > 0:
                                rms = np.sqrt(np.mean(audio_window**2))
                                
                                if rms > 0.015:
                                    print("█", end="", flush=True)
                                elif rms > 0.008:
                                    print("▓", end="", flush=True)
                                elif rms > 0.003:
                                    print("▒", end="", flush=True)
                                else:
                                    print(".", end="", flush=True)
                    
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"\nRecording error: {e}")
                self.is_recording = False
        
        self.recording_thread = threading.Thread(target=record_thread, daemon=True)
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording and save audio file."""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        try:
            # Stop recording
            sd.stop()
            
            # Wait for recording thread to finish
            if self.recording_thread:
                self.recording_thread.join(timeout=1.0)
            
            print("\nRecording stopped")
            
            if self.audio_data is None or len(self.audio_data) == 0:
                print("No audio recorded")
                return None
            
            # Trim silence from end (keep only recorded portion)
            # Find last non-silent frame
            rms_threshold = 0.001
            last_sound_idx = len(self.audio_data)
            
            for i in range(len(self.audio_data) - 1, 0, -int(0.1 * AUDIO_SAMPLE_RATE)):
                window = self.audio_data[max(0, i - int(0.1 * AUDIO_SAMPLE_RATE)):i]
                if len(window) > 0:
                    rms = np.sqrt(np.mean(window**2))
                    if rms > rms_threshold:
                        last_sound_idx = i
                        break
            
            # Trim to actual recording
            audio_data = self.audio_data[:last_sound_idx]
            
            if len(audio_data) == 0:
                print("No audio detected")
                return None
            
            # ========== DEBUG: Audio info ==========
            duration = len(audio_data) / AUDIO_SAMPLE_RATE
            max_amplitude_raw = np.max(np.abs(audio_data))
            rms_raw = np.sqrt(np.mean(audio_data**2))
            
            print(f"\n🔍 [DEBUG] Audio Recording Analysis:")
            print(f"   Total samples: {len(audio_data):,}")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Max amplitude (raw): {max_amplitude_raw:.4f}")
            print(f"   RMS (raw): {rms_raw:.4f}")
            
            if duration < 0.3:
                print(f"   ⚠️  WARNING: Audio muy corto ({duration:.2f}s)")
            
            if max_amplitude_raw < 0.01:
                print(f"   ⚠️  WARNING: Audio muy bajo (max: {max_amplitude_raw:.4f})")
            # ========================================
            
            # ========== NORMALIZACIÓN ==========
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normalize volume
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0:
                audio_data = audio_data * (0.9 / max_amplitude)
            
            print(f"   Max amplitude (normalized): {np.max(np.abs(audio_data)):.4f}")
            # ===================================
            
            # Save to WAV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = Path(f"/tmp/biobot_audio_{timestamp}.wav")
            
            # Convert to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(str(audio_path), 'wb') as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            
            file_size = audio_path.stat().st_size / 1024
            print(f"\n✓ Audio saved: {audio_path}")
            print(f"   File size: {file_size:.1f} KB")
            
            # ========== DEBUG: Verify WAV ==========
            try:
                with wave.open(str(audio_path), 'rb') as wf:
                    print(f"\n🔍 [DEBUG] WAV File Verification:")
                    print(f"   Channels: {wf.getnchannels()}")
                    print(f"   Sample rate: {wf.getframerate()} Hz")
                    print(f"   Duration: {wf.getnframes() / wf.getframerate():.2f}s")
            except Exception as e:
                print(f"   ⚠️  Could not verify WAV: {e}")
            # =======================================
            
            return audio_path
            
        except Exception as e:
            error_msg = f"Error stopping recording: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            if DASHBOARD:
                DASHBOARD.update_stats(last_error=error_msg)
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
                
                if DASHBOARD:
                    DASHBOARD.set_processing_step("Transcribing (local)", 20)

                # Load model (cached after first run)
                model = WhisperModel(
                    LOCAL_WHISPER_MODEL, 
                    device="cpu", 
                    compute_type="int8", 
                    num_workers=4, 
                    download_root=None
                )
                
                print(f"\n🔍 [DEBUG] Pre-Transcription Check:")
                audio_file = Path(audio_path)
                if not audio_file.exists():
                    print(f"   ❌ File does not exist: {audio_path}")
                    if DASHBOARD:
                        DASHBOARD.update_stats(last_error="Audio file not found")
                        DASHBOARD.set_processing_step(None, 0)
                    return None

                file_size = audio_file.stat().st_size
                print(f"   File: {audio_path}")
                print(f"   Size: {file_size / 1024:.1f} KB")

                # Verify WAV file is valid
                try:
                    import wave
                    with wave.open(audio_path, 'rb') as wf:
                        print(f"   Channels: {wf.getnchannels()}")
                        print(f"   Sample rate: {wf.getframerate()} Hz")
                        print(f"   Duration: {wf.getnframes() / wf.getframerate():.2f}s")
                        
                        # Check if duration is too short
                        wav_duration = wf.getnframes() / wf.getframerate()
                        if wav_duration < 0.3:
                            print(f"   ⚠️  WARNING: Audio muy corto para transcribir")
                except Exception as e:
                    print(f"   ⚠️  Invalid WAV file: {e}")
                # ===================================================================

                # Transcribe
                print(f"\n🔊 Starting Whisper transcription...")
                segments, info = model.transcribe(
                    audio_path, 
                    beam_size=5, 
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,  # ← BAJADO de 500 a 300
                        threshold=0.3  # ← Ya estaba en 0.3
                    ),
                    temperature=0.0, 
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.8
                )

                print(f"✓ Whisper finished, processing segments...")
                
                # Extraer texto de segmentos
                transcript_parts = []
                for segment in segments:
                    text = segment.text.strip()
                    if text:  # Only add non-empty segments
                        transcript_parts.append(text)
                        # Debug: mostrar confianza de cada segmento
                        print(f"   [{segment.start:.1f}s - {segment.end:.1f}s] {text}")
                
                transcript = " ".join(transcript_parts)
                
                # Detected language information
                print(f"   Detected language: {info.language} (probability: {info.language_probability:.2f})")
                print(f"   Duration: {info.duration:.1f}s")
                
                if not transcript:
                    print("⚠️  No se detectó voz en el audio")
                    if DASHBOARD:
                        DASHBOARD.update_stats(
                            last_error="No speech detected in audio",
                            mode="Idle"
                        )
                        DASHBOARD.set_processing_step(None, 0)  # ← LIMPIAR AQUÍ
                    return None
                
                print(f"Transcription: \"{transcript}\"")
                
                # ← LIMPIAR DASHBOARD ANTES DE RETORNAR ÉXITO
                if DASHBOARD:
                    DASHBOARD.set_processing_step(None, 0)
                
                return transcript.strip()
                
            except ImportError:
                print("faster-whisper not installed, falling back to API...")
                print("   Install with: pip install faster-whisper")
                if DASHBOARD:
                    DASHBOARD.update_stats(last_error="faster-whisper not installed")
                    DASHBOARD.set_processing_step(None, 0)  # ← LIMPIAR
                # Fall through to API options
            except Exception as e:
                error_msg = f"Local Whisper error: {e}"
                print(error_msg)
                if DASHBOARD:
                    DASHBOARD.update_stats(last_error=error_msg, mode="Idle")
                    DASHBOARD.set_processing_step(None, 0)  # ← LIMPIAR
                # Fall through to API options
        
        # Option 2: Groq API - FREE (for now) & FAST
        if USE_GROQ_STT and GROQ_API_KEY:
            print("   Using Groq Whisper API...")
            
            if DASHBOARD:
                DASHBOARD.set_processing_step("Transcribing (Groq)", 20)
            
            client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_file,
                    response_format="verbose_json",
                    language="en",
                    temperature=0.0
                )
            
            transcript_text = transcript.text if hasattr(transcript, 'text') else str(transcript)
            print(f"Transcription: \"{transcript_text}\"")
            
            if DASHBOARD:
                DASHBOARD.set_processing_step(None, 0)  # ← LIMPIAR
            
            return transcript_text.strip()
        
        # Option 3: OpenAI API - PAID but RELIABLE
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your-"):
            print("   Using OpenAI Whisper API...")
            
            if DASHBOARD:
                DASHBOARD.set_processing_step("Transcribing (OpenAI)", 20)
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=OPENAI_STT_MODEL,
                    file=audio_file,
                    response_format="text"
                )
            
            print(f"Transcription: \"{transcript}\"")
            
            if DASHBOARD:
                DASHBOARD.set_processing_step(None, 0)  # ← LIMPIAR
            
            return transcript.strip()
        
        # No valid STT option configured
        print("No STT service configured!")
        print("   Enable one of:")
        print("   - USE_LOCAL_WHISPER = True (free, offline)")
        print("   - USE_GROQ_STT = True with GROQ_API_KEY")
        print("   - Set OPENAI_API_KEY")
        
        if DASHBOARD:
            DASHBOARD.update_stats(last_error="No STT service configured", mode="Idle")
            DASHBOARD.set_processing_step(None, 0)  # ← LIMPIAR
        
        return None
        
    except Exception as e:
        error_msg = f"Error transcribing audio: {e}"
        print(error_msg)
        if DASHBOARD:
            DASHBOARD.update_stats(last_error=error_msg, mode="Idle")
            DASHBOARD.set_processing_step(None, 0)  # ← LIMPIAR
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


# ============ KNOWLEDGE BASE FUNCTIONS ============

def list_knowledge_bases() -> List[Dict[str, Any]]:
    """List all available knowledge bases from Open WebUI.
    
    Uses: GET /api/v1/knowledge/
    
    Returns:
        List of knowledge bases with id, name, description
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        url = f"{OPENWEBUI_URL}/api/v1/knowledge/"
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
        
        result = response.json()
        knowledge_bases = result.get("items", [])
        
        if DASHBOARD:
            DASHBOARD.add_debug_log(f"Found {len(knowledge_bases)} knowledge bases", "info")
        
        return knowledge_bases
        
    except Exception as e:
        print(f"Error listing knowledge bases: {e}")
        if DASHBOARD:
            DASHBOARD.add_debug_log(f"Error listing KBs: {e}", "error")
        return []


def get_knowledge_base_by_id(kb_id: str) -> Optional[Dict[str, Any]]:
    """Get knowledge base details by ID.
    
    Uses: GET /api/v1/knowledge/{id}
    
    Args:
        kb_id: Knowledge base ID
        
    Returns:
        Knowledge base details or None if not found
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        url = f"{OPENWEBUI_URL}/api/v1/knowledge/{kb_id}"
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        print(f"Error getting knowledge base {kb_id}: {e}")
        if DASHBOARD:
            DASHBOARD.add_debug_log(f"Error getting KB {kb_id}: {e}", "error")
        return None


def set_knowledge_base(kb_id: str, kb_name: str = None):
    """Set the current active knowledge base.
    
    Args:
        kb_id: Knowledge base ID to activate
        kb_name: Optional name (will be fetched if not provided)
    """
    global CURRENT_KNOWLEDGE_BASE_ID, CURRENT_KNOWLEDGE_BASE_NAME, DASHBOARD
    
    CURRENT_KNOWLEDGE_BASE_ID = kb_id
    
    if kb_name is None:
        # Fetch the name
        kb_details = get_knowledge_base_by_id(kb_id)
        if kb_details:
            kb_name = kb_details.get("name", "Unknown")
    
    CURRENT_KNOWLEDGE_BASE_NAME = kb_name
    
    print(f"✅ Knowledge base set: {kb_name} ({kb_id})")
    
    if DASHBOARD:
        DASHBOARD.add_debug_log(f"KB activated: {kb_name}", "success")
        DASHBOARD.update_stats(knowledge_base=kb_name)


def create_chat_in_openwebui(model: str, system_prompt: Optional[str] = None) -> Optional[str]:
    """Create a new chat in Open WebUI and return the chat ID."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        chat_data = {
            "name": f"BioBot - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "models": [model]
        }
        
        # Add system prompt if provided
        if system_prompt:
            chat_data["params"] = {"system": system_prompt}
            print(f"   📝 System prompt: {system_prompt[:50]}...")
        
        payload = {"chat": chat_data}
        
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
        
        # DEBUG: Verificar variable global
        if DASHBOARD:
            DASHBOARD.add_debug_log(f"SYSTEM_PROMPT en query: '{SYSTEM_PROMPT}'", "info")
        
        # Choose model based on whether we have a screenshot
        model_to_use = DEFAULT_MODEL if screenshot_path else TEXT_ONLY_MODEL
        
        # Update dashboard with current model
        if DASHBOARD:
            DASHBOARD.update_stats(current_model=model_to_use)
        
        # Build messages array with system prompt
        messages = []
        
        # Always include BASE_SYSTEM_PROMPT + optional custom SYSTEM_PROMPT
        combined_prompt = BASE_SYSTEM_PROMPT
        
        if SYSTEM_PROMPT:
            # Append custom prompt to base
            combined_prompt = f"{BASE_SYSTEM_PROMPT}\n\nADDITIONAL INSTRUCTIONS:\n{SYSTEM_PROMPT}"
            if DASHBOARD:
                DASHBOARD.add_debug_log(f"✅ Base + Custom prompt", "success")
            print(f"   ✅ Using BASE + custom system prompt")
        else:
            if DASHBOARD:
                DASHBOARD.add_debug_log("ℹ️  Using BASE system prompt only", "info")
            print(f"   ℹ️  Using BASE system prompt (no custom prompt set)")
        
        messages.append({
            "role": "system",
            "content": combined_prompt
        })
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
            print(f"   📜 Including {len(conversation_history)} previous messages")
        
        # Add current user message with inline base64 if image
        if screenshot_path:
            image_base64 = encode_image_to_base64(screenshot_path)
            vision_system_prompt = (
                "You are a visual AI assistant. You can see and analyze images. "
                "When the user asks about 'this' or 'the image', refer to the screenshot they've shared. "
                "Describe what you see in detail and answer their questions about the visual content. "
                "Be direct and specific about what's visible in the image."
            )
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": query_text},
                    {"type": "text", "text": vision_system_prompt},
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
            "stream": False,
            "temperature": 0.7
        }
        
        # Add knowledge base (RAG) if configured
        if CURRENT_KNOWLEDGE_BASE_ID:
            payload["files"] = [
                {
                    "type": "collection",
                    "id": CURRENT_KNOWLEDGE_BASE_ID,
                    "name": CURRENT_KNOWLEDGE_BASE_NAME or "Knowledge Base"
                }
            ]
            print(f"   📚 Using knowledge base: {CURRENT_KNOWLEDGE_BASE_NAME}")
            if DASHBOARD:
                DASHBOARD.add_debug_log(f"RAG enabled: {CURRENT_KNOWLEDGE_BASE_NAME}", "success")
        
        # Send request (only to get response, NOT for persistence)
        url = f"{OPENWEBUI_URL}/api/chat/completions"
        print(f"   Using model: {model_to_use}")

        if DASHBOARD:
            DASHBOARD.set_processing_step("Querying model", 70)
        
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
        error_msg = f"HTTP Error {e.response.status_code}: {e.response.text}"
        if DASHBOARD:
            DASHBOARD.update_stats(last_error=error_msg)
        return None
    except httpx.RequestError as e:
        error_msg = f"Request Error: {e}"
        if DASHBOARD:
            DASHBOARD.update_stats(last_error=error_msg)
        print("   Is Open WebUI running? Check the URL in config.py")
        return None
    except Exception as e:
        error_msg = f"Error querying Open WebUI: {e}"
        if DASHBOARD:
            DASHBOARD.update_stats(last_error=error_msg)
        return None


def query_standalone(query_text: str, screenshot_path: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Query AI model directly using TotalGPT API (standalone mode, no OpenWebUI).
    
    This mode:
    - Uses TotalGPT API for completions
    - No chat persistence
    - No knowledge base support
    - Ideal for quick questions
    """
    global DASHBOARD
    
    try:
        print("Querying model (standalone mode)...")
        
        if DASHBOARD:
            DASHBOARD.add_debug_log("Standalone mode: Using TotalGPT API", "info")
        
        # Choose model based on whether we have a screenshot
        model_to_use = STANDALONE_VISION_MODEL if screenshot_path else STANDALONE_MODEL
        
        # Update dashboard with current model
        if DASHBOARD:
            DASHBOARD.update_stats(current_model=f"{model_to_use} (standalone)")
        
        # Build messages array with system prompt
        messages = []
        
        # Always include BASE_SYSTEM_PROMPT + optional custom SYSTEM_PROMPT
        combined_prompt = BASE_SYSTEM_PROMPT
        
        if SYSTEM_PROMPT:
            combined_prompt = f"{BASE_SYSTEM_PROMPT}\n\nADDITIONAL INSTRUCTIONS:\n{SYSTEM_PROMPT}"
            if DASHBOARD:
                DASHBOARD.add_debug_log("BASE + Custom prompt", "success")
            print(f"   ✅ Using BASE + custom system prompt")
        else:
            if DASHBOARD:
                DASHBOARD.add_debug_log("Using BASE system prompt only", "info")
            print(f"   ℹ️  Using BASE system prompt")
        
        messages.append({
            "role": "system",
            "content": combined_prompt
        })
        
        # Add conversation history (if keeping in memory for session)
        if conversation_history:
            messages.extend(conversation_history)
            print(f"   📜 Including {len(conversation_history)} previous messages (session only)")
        
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
            print(f"   📸 Included screenshot")
        else:
            messages.append({
                "role": "user",
                "content": query_text
            })
        
        # Prepare the API request to TotalGPT
        headers = {
            "Authorization": f"Bearer {TOTALGPT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # Send request to TotalGPT API
        url = "https://api.totalgpt.ai/v1/chat/completions"
        print(f"   Using model: {model_to_use}")
        
        if DASHBOARD:
            DASHBOARD.set_processing_step("Querying TotalGPT API", 70)
        
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
        
        # Return simplified result (compatible with display_response)
        return {
            "content": content,
            "_query_text": query_text,
            "_has_screenshot": screenshot_path is not None,
            "_standalone_mode": True
        }
        
    except httpx.HTTPStatusError as e:
        error_msg = f"TotalGPT API Error {e.response.status_code}: {e.response.text}"
        print(error_msg)
        if DASHBOARD:
            DASHBOARD.update_stats(last_error=error_msg)
            DASHBOARD.add_debug_log(error_msg, "error")
        return None
    except httpx.RequestError as e:
        error_msg = f"Request Error: {e}"
        print(error_msg)
        if DASHBOARD:
            DASHBOARD.update_stats(last_error=error_msg)
        return None
    except Exception as e:
        error_msg = f"Error querying TotalGPT: {e}"
        print(error_msg)
        if DASHBOARD:
            DASHBOARD.update_stats(last_error=error_msg)
        return None


def cancel_tts():
    """Cancel currently playing TTS audio."""
    global TTS_PROCESS, DASHBOARD
    
    if TTS_PROCESS and TTS_PROCESS.poll() is None:
        # Process is still running
        try:
            TTS_PROCESS.terminate()
            TTS_PROCESS.wait(timeout=1.0)
            print("\n🔇 TTS cancelled")
            if DASHBOARD:
                DASHBOARD.add_debug_log("TTS cancelled by user", "warning")
            play_beep(frequency=600, duration=0.1)
            return True
        except Exception as e:
            print(f"Error cancelling TTS: {e}")
            return False
    else:
        print("\n⚠️  No TTS currently playing")
        return False


def text_to_speech(text: str) -> Optional[str]:
    """Convert text to speech using TotalGPT API and play it."""
    global DASHBOARD
    
    try:
        if not USE_TTS:
            if DASHBOARD:
                DASHBOARD.add_debug_log("TTS disabled in config", "warning")
            return None
            
        if not TOTALGPT_API_KEY:
            print("⚠️  TTS enabled but TOTALGPT_API_KEY not set!")
            if DASHBOARD:
                DASHBOARD.add_debug_log("TTS: API key missing", "error")
            return None
        
        print(f"🔊 Converting response to speech... (length: {len(text)} chars)")
        if DASHBOARD:
            DASHBOARD.add_debug_log(f"TTS: Converting {len(text)} chars", "info")
        
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
        
        if DASHBOARD:
            DASHBOARD.add_debug_log("TTS: Sending to TotalGPT API", "info")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.totalgpt.ai/v1/audio/speech",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
        
        if DASHBOARD:
            DASHBOARD.add_debug_log(f"TTS: Received {len(response.content)} bytes", "success")
        
        # Save audio to temp file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = Path(f"/tmp/biobot_tts_{timestamp}.mp3")
        
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        print(f"Speech generated, playing... ({audio_path})")
        print(f"💡 Double-click Right Cmd to cancel TTS")
        if DASHBOARD:
            DASHBOARD.add_debug_log("TTS: Playing audio (double-click cmd_r to cancel)", "info")
        
        # Play audio using afplay (macOS built-in) - save process for cancellation
        global TTS_PROCESS
        TTS_PROCESS = subprocess.Popen(["afplay", str(audio_path)])
        
        # Wait for completion or cancellation
        try:
            TTS_PROCESS.wait()
            print("Speech playback complete")
            if DASHBOARD:
                DASHBOARD.add_debug_log("TTS: Playback complete", "success")
        except:
            print("Speech playback cancelled")
            if DASHBOARD:
                DASHBOARD.add_debug_log("TTS: Cancelled", "warning")
        finally:
            TTS_PROCESS = None
            # Clean up
            try:
                audio_path.unlink()
            except:
                pass
        
        return str(audio_path)
        
    except httpx.HTTPStatusError as e:
        error_msg = f"TTS API Error {e.response.status_code}: {e.response.text}"
        print(error_msg)
        if DASHBOARD:
            DASHBOARD.add_debug_log(error_msg, "error")
        return None
    except Exception as e:
        error_msg = f"TTS Error: {e}"
        print(error_msg)
        if DASHBOARD:
            DASHBOARD.add_debug_log(error_msg, "error")
        return None


def extract_summary(content: str) -> str:
    """Extract only the SUMMARY portion from the response for TTS.
    
    The model is prompted to format responses as:
    SUMMARY: [brief answer]
    DETAILED EXPLANATION: [long answer]
    
    This function extracts just the SUMMARY part.
    """
    global DASHBOARD
    
    try:
        print(f"[extract_summary] Content length: {len(content)} chars")
        print(f"[extract_summary] Looking for 'SUMMARY:' in content...")
        
        # Look for "SUMMARY:" marker
        if "SUMMARY:" in content:
            print(f"[extract_summary] Found 'SUMMARY:' marker!")
            # Extract from SUMMARY: to DETAILED EXPLANATION: (or end if not found)
            summary_start = content.find("SUMMARY:") + len("SUMMARY:")
            
            # Find where detailed explanation starts
            detail_markers = ["DETAILED EXPLANATION:", "EXPLANATION:", "DETAILS:", "\n\n"]
            detail_start = len(content)  # Default to end
            
            for marker in detail_markers:
                idx = content.find(marker, summary_start)
                if idx != -1:
                    detail_start = min(detail_start, idx)
            
            summary = content[summary_start:detail_start].strip()
            print(f"[extract_summary] Extracted summary: {summary[:100]}...")
            if DASHBOARD:
                DASHBOARD.add_debug_log(f"Extracted summary: {len(summary)} chars", "success")
            return summary
        
        # Fallback: If no SUMMARY marker, take first paragraph
        print(f"[extract_summary] No 'SUMMARY:' found, using first paragraph")
        if DASHBOARD:
            DASHBOARD.add_debug_log("No SUMMARY marker, using fallback", "warning")
        
        paragraphs = content.split("\n\n")
        if paragraphs:
            fallback = paragraphs[0].strip()
            print(f"[extract_summary] Using first paragraph: {fallback[:100]}...")
            return fallback
        
        # Last resort: first 3 sentences
        print(f"[extract_summary] Using first 3 sentences fallback")
        sentences = content.split(". ")
        return ". ".join(sentences[:3]) + "." if len(sentences) > 3 else content
        
    except Exception as e:
        print(f"Error extracting summary: {e}")
        if DASHBOARD:
            DASHBOARD.add_debug_log(f"Error extracting summary: {e}", "error")
        return content[:500]  # Fallback to first 500 chars


def display_response(response: Dict[str, Any]):
    """Display the response from model and optionally read summary via TTS."""
    global DASHBOARD
    
    try:
        # Extract content (simplified structure)
        content = response.get('content', '')
        
        if content:
            print("\n" + "="*60)
            print("BIOBOT RESPONSE:")
            print("="*60)
            print(content)
            print("="*60)
            
            # Text-to-speech: Only read the SUMMARY portion
            if USE_TTS:
                print(f"\n🔊 TTS is enabled, extracting summary...")
                if DASHBOARD:
                    DASHBOARD.add_debug_log("TTS enabled, extracting summary", "info")
                
                summary = extract_summary(content)
                print(f"📝 Summary extracted ({len(summary)} chars): {summary[:100]}...")
                
                if DASHBOARD:
                    DASHBOARD.add_debug_log(f"Summary: {summary[:50]}...", "info")
                
                print(f"\n🔊 Reading summary aloud...")
                text_to_speech(summary)
            else:
                print("\n⚠️  TTS is disabled (USE_TTS = False)")
                if DASHBOARD:
                    DASHBOARD.add_debug_log("TTS disabled in config", "warning")
        else:
            print("No content in response")
            
    except Exception as e:
        print(f"Error displaying response: {e}")


# ============ QWEN3 (TotalGPT) LLM CLASSIFIER/ENHANCER ============
QWEN3_MODEL = "Qwen-Qwen3-30B-A3B"  # Updated model name for TotalGPT
TOTALGPT_URL = "https://api.totalgpt.ai/v1/chat/completions"

COMMAND_LIST = [
    "set system prompt",
    "new chat",
    "change knowledge base",
    "list knowledge bases",
    # Future commands:
    # "enable knowledge base",
    # "disable knowledge base",
]

def classify_or_enhance_transcript(transcript: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Use Qwen3 via TotalGPT to classify transcript as command or query.
    If query, enhance it (clarify, enforce English, improve prompt).
    Returns dict: {"type": "command"|"query", "command": ..., "enhanced_query": ...}
    """
    system_prompt = (
        "You are a voice command and query router. "
        "If the user is giving a command, output type=command and the command. "
        "If the user is asking a question or making a request, output type=query and an improved, clarified, and English-only version of the query. "
        "Always output a JSON object with keys: type, command, enhanced_query. "
        f"Available commands: {', '.join(COMMAND_LIST)}. "
        "If context is provided, use it to improve the query."
    )
    user_prompt = transcript
    if context:
        user_prompt += f"\nContext: {context}"
    payload = {
        "model": QWEN3_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.2
    }
    headers = {
        "Authorization": f"Bearer {TOTALGPT_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(TOTALGPT_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            # Parse JSON from LLM output
            try:
                parsed = json.loads(content)
                return parsed
            except Exception:
                print("[Qwen3] Could not parse JSON from LLM output:", content)
                return {"type": "unknown", "raw": content}
    except Exception as e:
        print(f"[Qwen3] Error: {e}")
        return {"type": "error", "error": str(e)}


# ============ SYSTEM PROMPT MANAGEMENT ============
# System prompt is included in every /api/chat/completions request
# by adding a "system" role message at the beginning of the messages array.
# This is simpler and more efficient than recreating chats.



# ============ COMMAND ROUTER ============
def handle_voice_command(command: str, arg: Optional[str] = None):
    """Route recognized command to the correct function."""
    global SYSTEM_PROMPT, DASHBOARD, CURRENT_CHAT_ID, CONVERSATION_HISTORY
    global CURRENT_KNOWLEDGE_BASE_ID, CURRENT_KNOWLEDGE_BASE_NAME
    
    cmd = command.lower().strip()
    
    if cmd == "list knowledge bases":
        if not USE_OPENWEBUI:
            print("\n⚠️  Knowledge bases are only available in OpenWebUI mode")
            print("   Set USE_OPENWEBUI = True in config.py to enable")
            play_beep(frequency=300, duration=0.3)
            return
        
        print("\n" + "="*60)
        print("📚 COMMAND: List Knowledge Bases")
        print("="*60)
        
        kb_list = list_knowledge_bases()
        
        if kb_list:
            print(f"\n✅ Found {len(kb_list)} knowledge bases:")
            for i, kb in enumerate(kb_list, 1):
                kb_id = kb.get("id", "unknown")
                kb_name = kb.get("name", "Unnamed")
                kb_desc = kb.get("description", "No description")
                active = " [ACTIVE]" if kb_id == CURRENT_KNOWLEDGE_BASE_ID else ""
                print(f"\n   {i}. {kb_name}{active}")
                print(f"      ID: {kb_id}")
                print(f"      Description: {kb_desc}")
        else:
            print("\n⚠️  No knowledge bases found")
        
        print("="*60 + "\n")
        
        if DASHBOARD:
            DASHBOARD.update_stats(mode="Listed KBs")
        play_beep(frequency=800, duration=0.1)
    
    elif cmd.startswith("change knowledge base"):
        if not USE_OPENWEBUI:
            print("\n⚠️  Knowledge bases are only available in OpenWebUI mode")
            print("   Set USE_OPENWEBUI = True in config.py to enable")
            play_beep(frequency=300, duration=0.3)
            return
        
        print("\n" + "="*60)
        print("📚 COMMAND: Change Knowledge Base")
        print("="*60)
        
        if arg:
            # arg should be the KB name or ID
            kb_list = list_knowledge_bases()
            
            # Try to find by name (case insensitive) or ID
            target_kb = None
            for kb in kb_list:
                if (arg.lower() in kb.get("name", "").lower() or 
                    arg == kb.get("id", "")):
                    target_kb = kb
                    break
            
            if target_kb:
                set_knowledge_base(target_kb["id"], target_kb["name"])
                print(f"\n✅ Knowledge base changed to: {target_kb['name']}")
                play_beep(frequency=1000, duration=0.1)
                time.sleep(0.1)
                play_beep(frequency=1200, duration=0.1)
            else:
                print(f"\n⚠️  Knowledge base not found: {arg}")
                print("   Use 'list knowledge bases' to see available options")
                play_beep(frequency=300, duration=0.3)
        else:
            print("⚠️  No knowledge base specified")
            print("   Example: 'change knowledge base datacenter docs'")
            play_beep(frequency=300, duration=0.3)
        
        print("="*60 + "\n")
        
        if DASHBOARD:
            DASHBOARD.update_stats(mode="KB changed" if target_kb else "KB change failed")
    
    elif cmd.startswith("set system prompt"):
        print("\n" + "="*60)
        print("🎤 COMMAND: Set System Prompt")
        print("="*60)
        
        # Configure system prompt
        if arg:
            new_prompt = arg.strip()
            
            if DASHBOARD:
                DASHBOARD.add_debug_log(f"BEFORE: SYSTEM_PROMPT = '{SYSTEM_PROMPT}'", "info")
                DASHBOARD.add_debug_log(f"Assigning: '{new_prompt}'", "info")
            
            SYSTEM_PROMPT = new_prompt
            
            if DASHBOARD:
                DASHBOARD.add_debug_log(f"AFTER: SYSTEM_PROMPT = '{SYSTEM_PROMPT}'", "success")
            
            print(f"\n✅ System prompt configured:")
            print(f"   '{new_prompt}'")
            print(f"\n💡 System prompt will be applied automatically to each message")
            
            play_beep(frequency=1000, duration=0.1)
            time.sleep(0.1)
            play_beep(frequency=1200, duration=0.1)
            
            if DASHBOARD:
                DASHBOARD.update_stats(
                    mode="✅ System prompt set",
                    system_prompt=new_prompt
                )
        else:
            print("⚠️  No text provided for system prompt")
            print("   Example: 'set system prompt: you are an expert in datacenters'")
            play_beep(frequency=300, duration=0.3)
        
        print("="*60 + "\n")
    
    elif cmd == "new chat":
        print("\n" + "="*60)
        print("COMMAND: New Chat")
        print("="*60)
        create_new_chat()
        print("Chat reset successfully")
        print("="*60 + "\n")
        if DASHBOARD:
            DASHBOARD.update_stats(mode="New chat created")
        play_beep(frequency=1000, duration=0.1)
    
    else:
        print(f"⚠️  Command not recognized: {command}")
        if DASHBOARD:
            DASHBOARD.update_stats(mode="Unknown command")


# ============ MAIN PROCESSING FUNCTION ============
# Global variables to store conversation state
CONVERSATION_HISTORY = []  # Text-only for API calls to avoid huge payloads
CURRENT_CHAT_ID = None  # Open WebUI chat ID for frontend display
MODIFIER_KEYS_PRESSED = set()  # Track which modifier keys are currently pressed
TRIGGER_KEYS_PRESSED = set()  # Track which trigger keys are currently pressed
SYSTEM_PROMPT = None  # Optional custom system prompt (appends to BASE_SYSTEM_PROMPT)
CURRENT_KNOWLEDGE_BASE_ID = None  # Currently selected knowledge base ID
CURRENT_KNOWLEDGE_BASE_NAME = None  # Currently selected knowledge base name

# TTS cancellation support
TTS_PROCESS = None  # Current TTS subprocess (afplay)
LAST_CMD_R_PRESS_TIME = 0  # For double-click detection
DOUBLE_CLICK_THRESHOLD = 0.5  # seconds

# Base system prompt - ALWAYS included with all models
BASE_SYSTEM_PROMPT = """Structure your response in two parts:

1. SUMMARY: Start with a brief, clear summary (2-3 sentences max) that directly answers the question. This will be read aloud via text-to-speech.

2. DETAILED EXPLANATION: After the summary, provide your detailed reasoning, technical details, and comprehensive explanation in a separate paragraph.

Format example:
SUMMARY: [Your concise answer here]

DETAILED EXPLANATION:
[Your comprehensive response here with all technical details and reasoning]"""

# System prompt presets (these get appended to BASE_SYSTEM_PROMPT)
SYSTEM_PROMPT_PRESETS = {
    "datacenter": "You are an expert datacenter technician assistant. Focus on server hardware, networking, troubleshooting, and datacenter operations.",
    "debug": "You are a debugging assistant. Provide step-by-step explanations and break down complex topics into simple terms.",
    "brief": "Keep all responses extremely concise. Prioritize brevity in both summary and explanation.",
    "general": "You are a helpful AI assistant. Provide balanced, informative responses."
}




def create_new_chat():
    """Create a new chat (reset conversation)."""
    global CONVERSATION_HISTORY, CURRENT_CHAT_ID, DASHBOARD
    
    old_chat_id = CURRENT_CHAT_ID
    old_messages = len(CONVERSATION_HISTORY) // 2
    
    CONVERSATION_HISTORY = []
    CURRENT_CHAT_ID = None
    
    print(f"Cleared {old_messages} messages from conversation")
    if old_chat_id:
        print(f"Previous chat: {old_chat_id[:12]}...")
    
    if DASHBOARD:
        DASHBOARD.update_stats(
            current_chat_id=None,
            messages_sent=0,
            last_transcript="",
            last_response=""
        )
        DASHBOARD.add_debug_log(f"New chat created ({old_messages} msgs cleared)", "success")
    
    play_beep(frequency=1000, duration=0.1)


def configure_system_prompt():
    """Configure system prompt via text input or preset selection."""
    global SYSTEM_PROMPT, CURRENT_CHAT_ID, DASHBOARD
    
    # Temporarily pause live display
    # (User needs to interact with terminal)
    
    print("\n" + "="*60)
    print("System Prompt Configuration")
    print("="*60)
    print("\nPresets:")
    for i, (key, prompt) in enumerate(SYSTEM_PROMPT_PRESETS.items(), 1):
        print(f"  {i}. {key}: {prompt[:60]}...")
    print(f"  {len(SYSTEM_PROMPT_PRESETS) + 1}. custom (enter your own)")
    print(f"  {len(SYSTEM_PROMPT_PRESETS) + 2}. none (disable system prompt)")
    
    try:
        choice = input("\nSelect option (or press Enter to cancel): ").strip()
        
        if not choice:
            play_beep(frequency=400, duration=0.1)
            return
        
        choice_num = int(choice)
        preset_keys = list(SYSTEM_PROMPT_PRESETS.keys())
        
        if 1 <= choice_num <= len(preset_keys):
            selected_key = preset_keys[choice_num - 1]
            SYSTEM_PROMPT = SYSTEM_PROMPT_PRESETS[selected_key]
            print(f"\n✓ System prompt set to '{selected_key}' preset")
            
        elif choice_num == len(preset_keys) + 1:
            custom = input("\nEnter custom system prompt: ").strip()
            if custom:
                SYSTEM_PROMPT = custom
                print(f"\n✓ Custom system prompt set")
            else:
                play_beep(frequency=400, duration=0.1)
                return
                
        elif choice_num == len(preset_keys) + 2:
            SYSTEM_PROMPT = None
            print("\n✓ System prompt disabled")
        else:
            play_beep(frequency=300, duration=0.2)
            return
        
        # Reset chat and update dashboard
        CONVERSATION_HISTORY = []
        CURRENT_CHAT_ID = None
        
        if DASHBOARD:
            DASHBOARD.update_stats(
                system_prompt=SYSTEM_PROMPT,
                current_chat_id=None,
                messages_sent=0
            )
        
        print("\nChat reset - new system prompt will apply to next message")
        play_beep(frequency=1000, duration=0.1)
        
    except (ValueError, KeyboardInterrupt):
        play_beep(frequency=400, duration=0.1)
def process_query(audio_path: str, capture_screenshot_flag: bool, use_region_selection: bool = False):
    """Process the complete workflow: transcribe, optionally screenshot, query, display.
    
    Args:
        audio_path: Path to recorded audio file
        capture_screenshot_flag: Whether to capture screenshot
        use_region_selection: Whether to use region selector for screenshot
    """
    global CONVERSATION_HISTORY, CURRENT_CHAT_ID, DASHBOARD
    screenshot_path = None
    
    try:
        # Update dashboard: Processing
        if DASHBOARD:
            DASHBOARD.update_stats(mode="Processing...", recording=False)
        
        if not audio_path:
            print("No audio recorded, aborting...")
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Idle")
            return
        
        # Step 1: Transcribe audio
        if DASHBOARD:
            DASHBOARD.update_stats(mode="Transcribing...")
        
        message_text = transcribe_audio(audio_path)
        if not message_text:
            print("Transcription failed, aborting...")
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Idle")
            return
        
        # Update dashboard with transcript
        if DASHBOARD:
            DASHBOARD.update_stats(last_transcript=message_text)
        
        # Step 1.5: Classify/enhance with keyword matching
        if DASHBOARD:
            DASHBOARD.update_stats(mode="Analyzing command...")

        transcript_lower = message_text.lower().strip()

        # Detect "set system prompt" command (with argument extraction)
        if "set system prompt" in transcript_lower or "change system prompt" in transcript_lower:
            # Extract text after command
            if "set system prompt" in transcript_lower:
                keyword = "set system prompt"
            else:
                keyword = "change system prompt"
            
            idx = transcript_lower.find(keyword)
            arg = message_text[idx + len(keyword):].strip()
            
            # Limpiar conectores comunes (: to, etc.)
            if arg.startswith(":"):
                arg = arg[1:].strip()
            if arg.startswith("to"):
                arg = arg[2:].strip()
            
            if not arg:
                print("⚠️  No se proporcionó texto para el system prompt")
                print("   Ejemplo: 'set system prompt: eres un experto en datacenters'")
                play_beep(frequency=300, duration=0.3)
                if DASHBOARD:
                    DASHBOARD.update_stats(mode="Idle")
                return
            
            print(f"[Command] Detected: set system prompt")
            print(f"[Argument] '{arg}'")
            
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Applying system prompt...")
            
            handle_voice_command("set system prompt", arg)
            
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Idle")
            return

        # Detectar comando "new chat"
        if any(phrase in transcript_lower for phrase in ["new chat", "start new chat", "reset chat", "create new chat"]):
            print(f"[Command] Detected: new chat")
            print(f"[Transcript] '{message_text}'")
            
            if DASHBOARD:
                DASHBOARD.add_debug_log("Command: new chat", "info")
                DASHBOARD.update_stats(mode="Executing: new chat")
            
            handle_voice_command("new chat")
            
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Idle")
                DASHBOARD.add_debug_log("New chat created", "success")
            
            return

        # Detectar comando "list knowledge bases"
        if any(phrase in transcript_lower for phrase in ["list knowledge bases", "show knowledge bases", "list knowledge base"]):
            print(f"[Command] Detected: list knowledge bases")
            print(f"[Transcript] '{message_text}'")
            
            if DASHBOARD:
                DASHBOARD.add_debug_log("Command: list knowledge bases", "info")
                DASHBOARD.update_stats(mode="Executing: list KBs")
            
            handle_voice_command("list knowledge bases")
            
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Idle")
            
            return

        # Detectar comando "change knowledge base"
        if any(phrase in transcript_lower for phrase in ["change knowledge base", "switch knowledge base", "use knowledge base"]):
            print(f"[Command] Detected: change knowledge base")
            print(f"[Transcript] '{message_text}'")
            
            # Extract the KB name from the transcript
            # Pattern: "change knowledge base [to] <name>"
            kb_name = None
            for trigger in ["change knowledge base", "switch knowledge base", "use knowledge base"]:
                if trigger in transcript_lower:
                    parts = transcript_lower.split(trigger, 1)
                    if len(parts) > 1:
                        kb_name = parts[1].strip()
                        # Remove common filler words
                        kb_name = kb_name.replace(" to ", "").strip()
                        break
            
            if DASHBOARD:
                DASHBOARD.add_debug_log(f"Command: change KB to '{kb_name}'", "info")
                DASHBOARD.update_stats(mode="Executing: change KB")
            
            handle_voice_command("change knowledge base", kb_name)
            
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Idle")
            
            return

        # Note: "delete last" command removed - not working reliably

        # No es comando - continuar con query normal
        print(f"[Query] Processing: {message_text}")
        
        # Not a command - enhance query if needed (optional)
        # For now, just use the transcript as-is
        print(f"[Query] Processing: {message_text}")

        # Step 2: Capture screenshot if requested
        if capture_screenshot_flag:
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Capturing screenshot...")
            print("Capturing screenshot...")
            screenshot_path = capture_screenshot(use_region_selection)
            if not screenshot_path:
                print("Screenshot capture failed, continuing without image...")
        else:
            print("Skipping screenshot (audio-only mode)")
        
        # Step 3: Query model - Different path for OpenWebUI vs Standalone mode
        if USE_OPENWEBUI:
            # OpenWebUI Mode: Full featured with persistence
            # Step 3a: Create chat on first message
            if CURRENT_CHAT_ID is None:
                if DASHBOARD:
                    DASHBOARD.update_stats(mode="Creating chat...")
                CURRENT_CHAT_ID = create_chat_in_openwebui(DEFAULT_MODEL, SYSTEM_PROMPT)
                if not CURRENT_CHAT_ID:
                    print("Could not create chat")
                    if DASHBOARD:
                        DASHBOARD.update_stats(mode="Idle")
                    return
                if DASHBOARD:
                    DASHBOARD.update_stats(current_chat_id=CURRENT_CHAT_ID)
            
            # Step 3b: Upload image if we have one
            file_data = None
            if screenshot_path:
                if DASHBOARD:
                    DASHBOARD.update_stats(mode="Uploading image...")
                file_data = upload_image_to_openwebui(screenshot_path)
                if not file_data:
                    print("Image upload failed, continuing without image...")
            
            # Step 3c: Query OpenWebUI (inline base64 for quick response)
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Querying model...")
            
            response = query_openwebui(message_text, screenshot_path, CONVERSATION_HISTORY, None)
        else:
            # Standalone Mode: Direct TotalGPT API, no persistence
            print("\n📦 Standalone mode: Using TotalGPT API directly")
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Querying model (standalone)...")
                DASHBOARD.add_debug_log("Standalone mode active", "info")
            
            response = query_standalone(message_text, screenshot_path, CONVERSATION_HISTORY)
        
        if not response:
            print("No response from model")
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Idle")
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
            
            # Update dashboard with response
            if DASHBOARD:
                messages_count = DASHBOARD.stats["messages_sent"] + 1
                DASHBOARD.update_stats(
                    last_response=assistant_message,
                    messages_sent=messages_count
                )
            
            # Save to chat with attachments (OpenWebUI mode only)
            if USE_OPENWEBUI:
                if DASHBOARD:
                    DASHBOARD.update_stats(mode="Saving to chat...")
                add_message_to_chat(CURRENT_CHAT_ID, message_text, assistant_message, file_data if screenshot_path else None)
            
            play_submit_beep()  # Audio confirmation that message was submitted
        
        # Step 6: Display the response and play TTS if enabled
        display_response(response)
        
        # Update dashboard: Ready
        if DASHBOARD:
            DASHBOARD.update_stats(mode="Idle")
        
        print("\n✅ Query complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if DASHBOARD:
            DASHBOARD.update_stats(mode="Error")
    
    finally:
        # Save screenshot to permanent folder for debugging
        if screenshot_path and Path(screenshot_path).exists():
            try:
                screenshots_dir = Path(__file__).parent / "screenshots"
                screenshots_dir.mkdir(exist_ok=True)
                
                import shutil
                permanent_path = screenshots_dir / Path(screenshot_path).name
                shutil.copy2(screenshot_path, permanent_path)
            except Exception as e:
                print(f"Could not save screenshot: {e}")
        
        # Clean up temporary files
        if audio_path and Path(audio_path).exists():
            try:
                Path(audio_path).unlink()
            except Exception as e:
                print(f"Could not delete audio file: {e}")
        
        if screenshot_path and Path(screenshot_path).exists():
            try:
                Path(screenshot_path).unlink()
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
    """Handle key press events - start recording."""
    global MODIFIER_KEYS_PRESSED, TRIGGER_KEYS_PRESSED, CONVERSATION_HISTORY, CURRENT_CHAT_ID, DASHBOARD
    global LAST_CMD_R_PRESS_TIME, DOUBLE_CLICK_THRESHOLD
    
    # Use config defaults if not provided
    key1 = trigger_key_screenshot or TRIGGER_KEY_WITH_SCREENSHOT
    key2 = trigger_key_audio or TRIGGER_KEY_AUDIO_ONLY

    # Build trigger map once
    trigger_map = get_trigger_key_map(key1, key2)

    try:
        from pynput.keyboard import Key

        # ESC = Reset conversation
        if key == Key.esc:
            if CONVERSATION_HISTORY or CURRENT_CHAT_ID:
                print("\n" + "="*60)
                print("ESC: Conversation reset - Starting fresh...")
                print("="*60 + "\n")
                CONVERSATION_HISTORY.clear()
                CURRENT_CHAT_ID = None
            return

        # Check if this is a trigger key FIRST (before checking modifiers)
        is_key1 = check_key_match(key, key1, trigger_map)
        is_key2 = check_key_match(key, key2, trigger_map)

        # Track modifier keys that are NOT trigger keys
        if not is_key1 and not is_key2:
            if key in (Key.cmd, Key.cmd_r, Key.cmd_l):
                MODIFIER_KEYS_PRESSED.add('cmd')
                return
            elif key in (Key.shift, Key.shift_r, Key.shift_l):
                MODIFIER_KEYS_PRESSED.add('shift')
                return
            elif key in (Key.alt, Key.alt_r, Key.alt_l):
                MODIFIER_KEYS_PRESSED.add('alt')
                return
            elif key in (Key.ctrl, Key.ctrl_r, Key.ctrl_l):
                MODIFIER_KEYS_PRESSED.add('ctrl')
                return

        # If not a trigger key, ignore
        if not is_key1 and not is_key2:
            return

        # Prevent re-triggering if key already pressed
        key_id = str(key)
        if key_id in TRIGGER_KEYS_PRESSED:
            return
        TRIGGER_KEYS_PRESSED.add(key_id)

        # Already recording - ignore additional presses
        if recorder.is_recording:
            return

        # ========== KEY 1 COMBINATIONS (Screenshot + Audio) ==========
        if is_key1:
            # Check for double-click on cmd_r to cancel TTS
            current_time = time.time()
            if key1 == 'cmd_r' and key == Key.cmd_r:
                time_since_last = current_time - LAST_CMD_R_PRESS_TIME
                if time_since_last < DOUBLE_CLICK_THRESHOLD:
                    # Double click detected - cancel TTS
                    print(f"\n[Double Click Detected] Cancelling TTS...")
                    cancel_tts()
                    LAST_CMD_R_PRESS_TIME = 0  # Reset
                    return
                else:
                    # First click - update timestamp
                    LAST_CMD_R_PRESS_TIME = current_time
            
            # Ctrl + Key1 = Reserved for future use
            # if 'ctrl' in MODIFIER_KEYS_PRESSED:
            #     # Future feature
            #     return

            # Alt + Key1 = Create new chat
            if 'alt' in MODIFIER_KEYS_PRESSED:
                create_new_chat()
                return

            # Cmd (left) + Key1 = Region selection mode
            if 'cmd' in MODIFIER_KEYS_PRESSED:
                if DASHBOARD:
                    DASHBOARD.update_stats(mode="Region Selection", recording=True)
                recorder.use_region_selection = True
                recorder.capture_screenshot = True
                recorder.start_recording()
                return

            # Just Key1 = Screenshot + audio
            play_double_beep()
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Recording (Screenshot)", recording=True)
            recorder.use_region_selection = False
            recorder.capture_screenshot = True
            recorder.start_recording()

        # ========== KEY 2 COMBINATIONS (Audio Only) ==========
        elif is_key2:
            # Alt + Key2 = Configure system prompt
            if 'alt' in MODIFIER_KEYS_PRESSED:
                configure_system_prompt()
                return

            # Just Key2 = Audio only
            play_start_beep()
            if DASHBOARD:
                DASHBOARD.update_stats(mode="Recording (Audio Only)", recording=True)
            recorder.use_region_selection = False
            recorder.capture_screenshot = False
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
    global MODIFIER_KEYS_PRESSED, TRIGGER_KEYS_PRESSED
    
    # Use config defaults if not provided
    key1 = trigger_key_screenshot or TRIGGER_KEY_WITH_SCREENSHOT
    key2 = trigger_key_audio or TRIGGER_KEY_AUDIO_ONLY
    
    # Build trigger map once
    trigger_map = get_trigger_key_map(key1, key2)
    
    try:
        from pynput.keyboard import Key
        
        # DEBUG: Print key release info
        print(f"[DEBUG] Key released: {key}")
        
        # Check if either trigger key was released FIRST
        is_key1 = check_key_match(key, key1, trigger_map)
        is_key2 = check_key_match(key, key2, trigger_map)
        
        if is_key1 or is_key2:
            print(f"[DEBUG] Trigger key released, stopping recording...")
            
            # Remove from pressed set
            key_id = str(key)
            TRIGGER_KEYS_PRESSED.discard(key_id)
            
            if recorder.is_recording:
                play_stop_beep()  # Audio feedback when recording stops
                capture_screenshot_flag = recorder.capture_screenshot
                use_region_flag = recorder.use_region_selection
                audio_path = recorder.stop_recording()
                
                # Reset recorder state
                recorder.use_region_selection = False
                recorder.capture_screenshot = True
                
                if audio_path:
                    process_query(audio_path, capture_screenshot_flag, use_region_flag)
            else:
                print(f"[DEBUG] Recorder was not recording")
            return
        
        # Track modifier key releases (only if they're not trigger keys)
        if key in (Key.cmd, Key.cmd_r, Key.cmd_l):
            MODIFIER_KEYS_PRESSED.discard('cmd')
            print(f"[DEBUG] Cmd modifier released")
            return
        elif key in (Key.shift, Key.shift_r, Key.shift_l):
            MODIFIER_KEYS_PRESSED.discard('shift')
            print(f"[DEBUG] Shift modifier released")
            return
        elif key in (Key.alt, Key.alt_r, Key.alt_l):
            MODIFIER_KEYS_PRESSED.discard('alt')
            print(f"[DEBUG] Alt modifier released")
            return
        elif key in (Key.ctrl, Key.ctrl_r, Key.ctrl_l):
            MODIFIER_KEYS_PRESSED.discard('ctrl')
            print(f"[DEBUG] Ctrl modifier released")
            return
        
    except Exception as e:
        print(f"Error in key release handler: {e}")
        import traceback
        traceback.print_exc()



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
    global DASHBOARD, LIVE_DISPLAY
    
    # Initialize dashboard
    DASHBOARD = get_dashboard()
    
    # Check permissions
    if not check_all_permissions():
        print("\nExiting due to missing permissions.")
        return
    
    # Check for first-time setup
    custom_keys = check_first_run()
    
    if custom_keys:
        trigger_key_screenshot, trigger_key_audio = custom_keys
    else:
        trigger_key_screenshot = TRIGGER_KEY_WITH_SCREENSHOT
        trigger_key_audio = TRIGGER_KEY_AUDIO_ONLY
    
    # Update dashboard with initial config
    operation_mode = "OpenWebUI" if USE_OPENWEBUI else "Standalone"
    model_display = DEFAULT_MODEL if USE_OPENWEBUI else STANDALONE_MODEL
    
    DASHBOARD.update_stats(
        openwebui_url=f"📦 {operation_mode} Mode",
        stt_service="Local Whisper" if USE_LOCAL_WHISPER else ("Groq" if USE_GROQ_STT else "OpenAI"),
        tts_enabled=USE_TTS,
        current_model=model_display
    )
    
    print(f"\n{'='*60}")
    print(f"OPERATION MODE: {operation_mode}")
    print(f"{'='*60}")
    
    if USE_OPENWEBUI:
        # OpenWebUI Mode: Test connection and setup knowledge base
        print(f"Connecting to OpenWebUI at {OPENWEBUI_URL}...")
        
        try:
            with httpx.Client(timeout=10.0) as client:
                headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
                response = client.get(f"{OPENWEBUI_URL}/api/config", headers=headers)
                response.raise_for_status()
                DASHBOARD.update_stats(openwebui_url=f"✓ {OPENWEBUI_URL}")
                print(f"✅ Connected to OpenWebUI successfully")
        except Exception as e:
            DASHBOARD.update_stats(openwebui_url=f"✗ {OPENWEBUI_URL}")
            print(f"❌ Could not connect to Open WebUI: {e}")
            print(f"\n💡 Tip: Set USE_OPENWEBUI = False in config.py for standalone mode")
            exit(1)
        
        # Initialize knowledge base if configured in config.py
        if KNOWLEDGE_ID:
            print(f"\n🔄 Initializing knowledge base from config...")
            
            # Check if KNOWLEDGE_ID looks like a UUID (contains dashes) or a name
            if "-" in KNOWLEDGE_ID and len(KNOWLEDGE_ID) > 30:
                # It's likely a UUID, try to get it directly
                kb_details = get_knowledge_base_by_id(KNOWLEDGE_ID)
                if kb_details:
                    kb_name = kb_details.get("name", "Unknown")
                    set_knowledge_base(KNOWLEDGE_ID, kb_name)
                    print(f"✅ Knowledge base loaded: {kb_name}")
                else:
                    print(f"⚠️  Could not load knowledge base {KNOWLEDGE_ID}")
                    print("   You can change it later with 'change knowledge base' command")
            else:
                # It's a name, search for it in the list
                print(f"   Searching for knowledge base named: {KNOWLEDGE_ID}")
                kb_list = list_knowledge_bases()
                
                target_kb = None
                for kb in kb_list:
                    if KNOWLEDGE_ID.lower() in kb.get("name", "").lower():
                        target_kb = kb
                        break
                
                if target_kb:
                    set_knowledge_base(target_kb["id"], target_kb["name"])
                    print(f"✅ Knowledge base loaded: {target_kb['name']} (ID: {target_kb['id'][:12]}...)")
                else:
                    print(f"⚠️  Could not find knowledge base matching: {KNOWLEDGE_ID}")
                    print(f"   Available knowledge bases:")
                    for kb in kb_list:
                        print(f"      - {kb.get('name', 'Unnamed')}")
                    print("   You can change it later with 'change knowledge base' command")
    else:
        # Standalone Mode: No OpenWebUI, using TotalGPT API directly
        print(f"Using TotalGPT API directly (no OpenWebUI)")
        print(f"Model: {STANDALONE_MODEL}")
        print(f"Vision Model: {STANDALONE_VISION_MODEL}")
        print(f"\n💡 Note: Standalone mode has no chat persistence or knowledge bases")
        print(f"         Set USE_OPENWEBUI = True for full features")
        
        DASHBOARD.add_debug_log("Standalone mode active", "info")
    
    # Create audio recorder
    recorder = AudioRecorder()
    
    # Start live dashboard
    try:
        with Live(DASHBOARD.render(), refresh_per_second=2, screen=True) as live:
            LIVE_DISPLAY = live
            
            # Set up keyboard listener
            with keyboard.Listener(
                on_press=lambda key: on_press(key, recorder, trigger_key_screenshot, trigger_key_audio),
                on_release=lambda key: on_release(key, recorder, trigger_key_screenshot, trigger_key_audio)
            ) as listener:
                # Update dashboard in loop
                while True:
                    live.update(DASHBOARD.render())
                    time.sleep(0.5)
                    
    except KeyboardInterrupt:
        print("\n\nShutting down BioBot...")
        print("Goodbye! \n")


if __name__ == "__main__":
    main()
