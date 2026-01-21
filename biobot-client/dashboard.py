#!/usr/bin/env python3
"""
BioBot Cyberpunk Dashboard - Real-time status display
"""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from datetime import datetime
from typing import Dict, Any, Optional
import time

class BioBotDashboard:
    """Cyberpunk-themed dashboard for BioBot status."""
    
    def __init__(self):
        self.console = Console()
        self.stats = {
            "openwebui_url": "Not connected",
            "current_chat_id": None,
            "messages_sent": 0,
            "current_model": "None",
            "stt_service": "None",
            "tts_enabled": False,
            "recording": False,
            "mode": "Idle",
            "last_transcript": "",
            "last_response": "",
            "system_prompt": None,
            "processing_step": None,  # ← NUEVO
            "processing_progress": 0,  # ← NUEVO (0-100)
        }
        self.start_time = datetime.now()
        
    def create_ascii_logo(self) -> Text:
        """Create ASCII art logo."""
        logo = """
    ██████╗ ██╗ ██████╗ ██████╗  ██████╗ ████████╗
    ██╔══██╗██║██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝
    ██████╔╝██║██║   ██║██████╔╝██║   ██║   ██║   
    ██╔══██╗██║██║   ██║██╔══██╗██║   ██║   ██║   
    ██████╔╝██║╚██████╔╝██████╔╝╚██████╔╝   ██║   
    ╚═════╝ ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   
        """
        text = Text(logo, style="bold green")
        return Align.center(text)
    
    def create_header(self) -> Panel:
        """Create header panel."""
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]
        
        header_text = Text()
        header_text.append("▬" * 60 + "\n", style="green")
        header_text.append("  BIOBOT VOICE CLIENT ", style="bold bright_green")
        header_text.append("v2.0", style="dim green")
        header_text.append(" | ", style="green")
        header_text.append("DATACENTER AI ASSISTANT\n", style="bold cyan")
        header_text.append("▬" * 60 + "\n", style="green")
        header_text.append(f"  UPTIME: {uptime_str} ", style="yellow")
        header_text.append("| ", style="green")
        header_text.append(f"STATUS: ", style="white")
        
        if self.stats["recording"]:
            header_text.append("● RECORDING", style="bold red blink")
        else:
            header_text.append("● ONLINE", style="bold green")
        
        return Panel(header_text, border_style="green", padding=(0, 1))
    
    def create_connection_panel(self) -> Panel:
        """Create connection info panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan", width=20)
        table.add_column("Value", style="bright_white")
        
        # OpenWebUI URL
        table.add_row(
            "🌐 OpenWebUI",
            self.stats["openwebui_url"]
        )
        
        # Current Chat
        chat_display = self.stats["current_chat_id"][:16] + "..." if self.stats["current_chat_id"] else "No active chat"
        table.add_row(
            "💬 Chat ID",
            chat_display
        )
        
        # Messages sent
        table.add_row(
            "📊 Messages Sent",
            str(self.stats["messages_sent"])
        )
        
        # Current model
        table.add_row(
            "🤖 Model",
            self.stats["current_model"]
        )
        
        # STT Service
        table.add_row(
            "🎤 STT Service",
            self.stats["stt_service"]
        )
        
        # TTS Status
        tts_status = "Enabled" if self.stats["tts_enabled"] else "Disabled"
        table.add_row(
            "🔊 TTS",
            tts_status
        )
        
        return Panel(table, title="[bold green]⚡ CONNECTION STATUS", border_style="green")
    
    def create_keyboard_shortcuts(self) -> Panel:
        """Create keyboard shortcuts panel."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Shortcut", style="yellow", width=20)
        table.add_column("Action", style="bright_white")
        
        shortcuts = [
            ("Right Cmd", "Screenshot + Audio"),
            ("Cmd + Right Cmd", "Region Select + Audio"),
            ("Right Shift", "Audio Only"),
            ("Ctrl + Right Cmd", "Delete Last Message"),
            ("Alt + Right Cmd", "New Chat"),
            ("Alt + Right Shift", "System Prompt"),
            ("ESC", "Reset Conversation"),
        ]
        
        for shortcut, action in shortcuts:
            table.add_row(f"⌨️  {shortcut}", action)
        
        return Panel(table, title="[bold green]⚡ KEYBOARD SHORTCUTS", border_style="green")
    
    def create_voice_commands(self) -> Panel:
        """Create voice commands panel."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Command", style="magenta", width=25)
        table.add_column("Description", style="bright_white")
        
        commands = [
            ("new chat", "Start a new conversation"),
            ("delete last", "Remove last message"),
            ("set system prompt", "Change system prompt"),
            ("repeat", "Repeat last response"),
            ("retake screenshot", "Capture new screenshot"),
        ]
        
        for cmd, desc in commands:
            table.add_row(f"🎙️  {cmd}", desc)
        
        return Panel(table, title="[bold green]⚡ VOICE COMMANDS", border_style="green")
    
    def create_activity_panel(self) -> Panel:
        """Create recent activity panel."""
        content = Text()
        
        # Current mode
        content.append("MODE: ", style="cyan")
        mode_style = "bold red" if self.stats["recording"] else "green"
        content.append(f"{self.stats['mode']}\n\n", style=mode_style)
        
        # Last transcript
        if self.stats["last_transcript"]:
            content.append("LAST INPUT:\n", style="yellow")
            transcript_preview = self.stats["last_transcript"][:100]
            if len(self.stats["last_transcript"]) > 100:
                transcript_preview += "..."
            content.append(f"  {transcript_preview}\n\n", style="white")
        
        # Last response
        if self.stats["last_response"]:
            content.append("LAST OUTPUT:\n", style="yellow")
            response_preview = self.stats["last_response"][:100]
            if len(self.stats["last_response"]) > 100:
                response_preview += "..."
            content.append(f"  {response_preview}\n", style="white")
        
        # System prompt
        if self.stats["system_prompt"]:
            content.append("\nSYSTEM PROMPT:\n", style="cyan")
            prompt_preview = self.stats["system_prompt"][:80]
            if len(self.stats["system_prompt"]) > 80:
                prompt_preview += "..."
            content.append(f"  {prompt_preview}", style="dim white")
        
        return Panel(content, title="[bold green]⚡ ACTIVITY LOG", border_style="green")
    
    def create_progress_bar(self) -> Panel:
        """Create animated progress bar for processing steps."""
        if not self.stats["processing_step"]:
            # Idle state
            footer_text = Text()
            footer_text.append("  Press ", style="dim white")
            footer_text.append("Right Cmd", style="bold yellow")
            footer_text.append(" (screenshot) or ", style="dim white")
            footer_text.append("Right Shift", style="bold yellow")
            footer_text.append(" (audio only) to start", style="dim white")
            return Panel(footer_text, border_style="green", height=3)
        
        # Processing state
        step = self.stats["processing_step"]
        progress = self.stats["processing_progress"]
        
        # Progress bar characters
        bar_width = 50
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Step icon
        step_icons = {
            "Transcribing": "🎤",
            "Analyzing": "🔍",
            "Capturing screenshot": "📸",
            "Creating chat": "💬",
            "Uploading image": "⬆️",
            "Querying model": "🤖",
            "Saving to chat": "💾",
            "Processing": "⚙️",
        }
        icon = step_icons.get(step, "⚙️")
        
        # Build progress display
        content = Text()
        content.append(f"{icon} ", style="bold yellow")
        content.append(f"{step}... ", style="bold cyan")
        content.append(f"[{bar}] ", style="green")
        content.append(f"{progress}%", style="bold white")
        
        # Animated spinner
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = int(time.time() * 10) % len(spinner_frames)
        content.append(f" {spinner_frames[spinner_idx]}", style="bold yellow blink")
        
        return Panel(content, border_style="yellow", height=3)
    
    def create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="connection", size=12),
            Layout(name="activity")
        )
        
        layout["right"].split_column(
            Layout(name="keyboard", size=12),
            Layout(name="voice", size=10)
        )
        
        # Populate panels
        layout["header"].update(self.create_header())
        layout["connection"].update(self.create_connection_panel())
        layout["keyboard"].update(self.create_keyboard_shortcuts())
        layout["voice"].update(self.create_voice_commands())
        layout["activity"].update(self.create_activity_panel())
        
        # Footer with progress bar
        layout["footer"].update(self.create_progress_bar())
        
        return layout
    
    def update_stats(self, **kwargs):
        """Update dashboard statistics."""
        self.stats.update(kwargs)
    
    def set_processing_step(self, step: Optional[str], progress: int = 0):
        """Update processing step and progress.
        
        Args:
            step: Name of current processing step (None for idle)
            progress: Progress percentage (0-100)
        """
        self.stats["processing_step"] = step
        self.stats["processing_progress"] = progress
    
    def render(self) -> Layout:
        """Render the dashboard."""
        return self.create_layout()


# Singleton instance
_dashboard_instance = None

def get_dashboard() -> BioBotDashboard:
    """Get or create dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = BioBotDashboard()
    return _dashboard_instance
