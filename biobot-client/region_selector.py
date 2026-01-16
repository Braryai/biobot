#!/usr/bin/env python3
"""
Region Selector - Interactive screen region selection overlay
Uses tkinter for cross-platform compatibility
"""

import tkinter as tk
from typing import Optional, Tuple
import subprocess
from pathlib import Path
from datetime import datetime


class RegionSelector:
    """Interactive overlay for selecting screen regions."""
    
    def __init__(self):
        self.root = None
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selection = None
        
    def select_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Show overlay and let user select a region.
        
        Returns:
            Tuple of (x, y, width, height) or None if cancelled
        """
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.3)
        self.root.configure(bg='black')
        self.root.cursor = 'crosshair'
        
        # Create canvas
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        self.canvas = tk.Canvas(
            self.root,
            width=screen_width,
            height=screen_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Instructions
        instructions = self.canvas.create_text(
            screen_width // 2,
            50,
            text="Drag to select region • ESC to cancel • Click without drag for focused window",
            fill='white',
            font=('Arial', 16)
        )
        
        # Bind mouse events
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.root.bind('<Escape>', self.on_cancel)
        
        self.root.mainloop()
        
        return self.selection
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        self.start_x = event.x
        self.start_y = event.y
        
        # Create rectangle
        self.rect = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline='red',
            width=2
        )
    
    def on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.rect:
            # Update rectangle
            self.canvas.coords(
                self.rect,
                self.start_x,
                self.start_y,
                event.x,
                event.y
            )
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if self.start_x is None or self.start_y is None:
            self.on_cancel(None)
            return
        
        end_x = event.x
        end_y = event.y
        
        # Calculate selection bounds
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)
        
        # If too small (just a click), return None to use focused window
        if width < 10 and height < 10:
            self.selection = None
        else:
            self.selection = (x, y, width, height)
        
        self.root.quit()
        self.root.destroy()
    
    def on_cancel(self, event):
        """Handle ESC key - cancel selection."""
        self.selection = None
        self.root.quit()
        self.root.destroy()


def capture_region_screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
    """Capture screenshot of selected region or focused window.
    
    Args:
        region: Tuple of (x, y, width, height) or None for focused window
        
    Returns:
        Path to screenshot file or None if failed
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = Path(f"/tmp/biobot_screenshot_{timestamp}.png")
    
    try:
        if region:
            x, y, width, height = region
            print(f"   Capturing region: {width}x{height} at ({x}, {y})")
            
            # Use screencapture with region bounds
            result = subprocess.run(
                ["screencapture", "-x", "-R", f"{x},{y},{width},{height}", "-t", "png", str(screenshot_path)],
                capture_output=True,
                timeout=5
            )
        else:
            # No region selected - fall back to focused window
            print("   No region selected, capturing focused window...")
            
            try:
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
                        
                        if layer == 0 and width > 100 and height > 100:
                            target_window_id = window.get('kCGWindowNumber')
                            print(f"   Window ID: {target_window_id}")
                            print(f"   Window bounds: {width:.0f}x{height:.0f}")
                            break
                
                if target_window_id:
                    result = subprocess.run(
                        ["screencapture", "-x", "-o", "-l", str(target_window_id), "-t", "png", str(screenshot_path)],
                        capture_output=True,
                        timeout=5
                    )
                else:
                    # Fall back to full screen
                    print("   Falling back to full screen...")
                    result = subprocess.run(
                        ["screencapture", "-x", "-t", "png", str(screenshot_path)],
                        capture_output=True,
                        timeout=5
                    )
            except ImportError:
                # No PyObjC - use full screen
                print("   PyObjC not available, using full screen...")
                result = subprocess.run(
                    ["screencapture", "-x", "-t", "png", str(screenshot_path)],
                    capture_output=True,
                    timeout=5
                )
        
        # Verify screenshot was created
        if result.returncode != 0:
            print(f"   screencapture returned code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.decode()}")
            return None
        
        if not screenshot_path.exists():
            print("   Screenshot file was not created")
            return None
        
        file_size = screenshot_path.stat().st_size
        if file_size < 1000:
            print(f"   Screenshot file too small: {file_size} bytes")
            screenshot_path.unlink()
            return None
        
        size_mb = file_size / (1024 * 1024)
        print(f"   Screenshot captured: {size_mb:.2f} MB")
        
        return str(screenshot_path)
        
    except subprocess.TimeoutExpired:
        print("   Screenshot capture timed out")
        return None
    except Exception as e:
        print(f"   Screenshot capture failed: {e}")
        return None


if __name__ == "__main__":
    # Test the region selector
    print("Region Selector Test")
    print("=" * 60)
    
    selector = RegionSelector()
    region = selector.select_region()
    
    if region:
        x, y, w, h = region
        print(f"\nSelected region: {w}x{h} at ({x}, {y})")
        
        screenshot = capture_region_screenshot(region)
        if screenshot:
            print(f"Screenshot saved: {screenshot}")
        else:
            print("Screenshot failed")
    else:
        print("\nNo region selected (cancelled or click)")
        screenshot = capture_region_screenshot(None)
        if screenshot:
            print(f"Screenshot saved (focused window): {screenshot}")
