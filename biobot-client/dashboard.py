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
            "knowledge_base": None,
            "processing_step": None,
            "processing_progress": 0,
            "debug_logs": [],  # ← NUEVO: Lista de logs de debug
        }
        self.start_time = datetime.now()
        self.max_debug_logs = 10  # Máximo de logs a mostrar
        
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
            ("set system prompt", "Configure AI behavior"),
            ("new chat", "Start a new conversation"),
            ("change knowledge base", "Switch knowledge base"),
            ("list knowledge bases", "Show available KBs"),
        ]
        
        for cmd, desc in commands:
            table.add_row(f"🎙️  {cmd}", desc)
        
        return Panel(table, title="[bold green]⚡ VOICE COMMANDS", border_style="green")
    
    def create_activity_panel(self) -> Panel:
        """Create detailed activity panel with debugging info."""
        content = Text()
        
        # Current mode with timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        content.append(f"[{current_time}] ", style="dim white")
        content.append("MODE: ", style="cyan")
        mode_style = "bold red" if self.stats["recording"] else "green"
        content.append(f"{self.stats['mode']}\n", style=mode_style)
        
        # Processing step (if active)
        if self.stats.get("processing_step"):
            content.append("\n⚙️  PROCESSING: ", style="yellow")
            content.append(f"{self.stats['processing_step']}\n", style="bold cyan")
            
            # Progress bar
            progress = self.stats.get("processing_progress", 0)
            bar_width = 30
            filled = int(bar_width * progress / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            content.append(f"   [{bar}] {progress}%\n", style="green")
        
        # Last transcript with timestamp
        if self.stats.get("last_transcript"):
            content.append("\n📥 LAST INPUT:\n", style="yellow")
            content.append(f"   {self.stats['last_transcript']}\n", style="white")
        
        # Last response with timestamp
        if self.stats.get("last_response"):
            content.append("\n📤 LAST OUTPUT:\n", style="yellow")
            response_preview = self.stats["last_response"][:200]
            if len(self.stats["last_response"]) > 200:
                response_preview += "..."
            content.append(f"   {response_preview}\n", style="white")
        
        # System prompt
        if self.stats.get("system_prompt"):
            content.append("\n🔧 SYSTEM PROMPT:\n", style="cyan")
            prompt_preview = self.stats["system_prompt"][:100]
            if len(self.stats["system_prompt"]) > 100:
                prompt_preview += "..."
            content.append(f"   {prompt_preview}\n", style="dim white")
        
        if self.stats.get("knowledge_base"):
            content.append("\n📚 KNOWLEDGE BASE:\n", style="cyan")
            content.append(f"   {self.stats['knowledge_base']}\n", style="dim white")
        
        # Debug info
        content.append("\n" + "─" * 50 + "\n", style="dim white")
        content.append("🐛 DEBUG INFO:\n", style="dim cyan")
        
        # Chat info
        if self.stats.get("current_chat_id"):
            chat_id_short = self.stats["current_chat_id"][:12] + "..."
            content.append(f"   Chat: {chat_id_short}\n", style="dim white")
        else:
            content.append(f"   Chat: None\n", style="dim red")
    
        # Message count
        content.append(f"   Messages: {self.stats.get('messages_sent', 0)}\n", style="dim white")
        
        # Model info
        content.append(f"   Model: {self.stats.get('current_model', 'N/A')}\n", style="dim white")
        
        # Last error (if any)
        if self.stats.get("last_error"):
            content.append(f"\n❌ LAST ERROR:\n", style="red")
            content.append(f"   {self.stats['last_error']}\n", style="dim red")
        
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
    
    def create_debug_panel(self) -> Panel:
        """Create debug logs panel."""
        content = Text()
        content.append("🐛 DEBUG LOGS\n", style="bold yellow")
        content.append("─" * 50 + "\n", style="dim white")
        
        if not self.stats["debug_logs"]:
            content.append("  No debug logs yet...\n", style="dim white")
        else:
            # Show logs in reverse order (newest first)
            for log in reversed(self.stats["debug_logs"]):
                time_str = log["time"]
                message = log["message"]
                level = log["level"]
                
                # Color based on level
                if level == "error":
                    style = "bold red"
                    icon = "❌"
                elif level == "warning":
                    style = "bold yellow"
                    icon = "⚠️ "
                elif level == "success":
                    style = "bold green"
                    icon = "✅"
                else:  # info
                    style = "white"
                    icon = "ℹ️ "
                
                content.append(f"[{time_str}] {icon} ", style="dim cyan")
                content.append(f"{message}\n", style=style)
        
        return Panel(content, title="[bold yellow]🔍 DEBUG", border_style="yellow")
    
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
            Layout(name="debug", size=15),  # ← NUEVO: Panel de debug
            Layout(name="voice", size=10)
        )
        
        # Populate panels
        layout["header"].update(self.create_header())
        layout["connection"].update(self.create_connection_panel())
        layout["keyboard"].update(self.create_keyboard_shortcuts())
        layout["debug"].update(self.create_debug_panel())  # ← NUEVO
        layout["voice"].update(self.create_voice_commands())
        layout["activity"].update(self.create_activity_panel())
        
        # Footer with progress bar
        layout["footer"].update(self.create_progress_bar())
        
        return layout
    
    def update_stats(self, **kwargs):
        """Update dashboard statistics."""
        self.stats.update(kwargs)
    
    def add_debug_log(self, message: str, level: str = "info"):
        """Add a debug log message to the dashboard.
        
        Args:
            message: The debug message
            level: Log level (info, warning, error, success)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "time": timestamp,
            "message": message,
            "level": level
        }
        self.stats["debug_logs"].append(log_entry)
        
        # Keep only last N logs
        if len(self.stats["debug_logs"]) > self.max_debug_logs:
            self.stats["debug_logs"] = self.stats["debug_logs"][-self.max_debug_logs:]
    
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
