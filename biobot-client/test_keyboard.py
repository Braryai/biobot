#!/usr/bin/env python3
"""
Test keyboard listener - to debug why biobot_voice.py doesn't respond
"""

from pynput import keyboard

print("="*60)
print("KEYBOARD LISTENER TEST")
print("="*60)
print("\nPress any keys to see if they are detected...")
print("Press 'q' to quit\n")

def on_press(key):
    """Print when any key is pressed."""
    try:
        if hasattr(key, 'char') and key.char == 'q':
            print("\n✓ Quit requested")
            return False  # Stop listener
        
        # Print key info
        if hasattr(key, 'char'):
            print(f"✓ Character key pressed: '{key.char}'")
        elif hasattr(key, 'name'):
            print(f"✓ Special key pressed: {key.name}")
        else:
            print(f"✓ Key pressed: {key}")
            
        # Check for cmd_r specifically
        if key == keyboard.Key.cmd_r:
            print("  → This is RIGHT COMMAND (cmd_r) ✓")
        elif key == keyboard.Key.cmd:
            print("  → This is LEFT COMMAND (cmd) ✓")
            
    except Exception as e:
        print(f"Error: {e}")

def on_release(key):
    """Print when any key is released."""
    try:
        if hasattr(key, 'char'):
            print(f"  Released: '{key.char}'")
        elif hasattr(key, 'name'):
            print(f"  Released: {key.name}")
    except:
        pass

# Start listener
print("Listening for keyboard events...")
print("Try pressing:")
print("  - Right Command key (cmd_r)")
print("  - Left Command key (cmd)")  
print("  - Any other keys")
print("\nPress 'q' to quit\n")

try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
except KeyboardInterrupt:
    print("\n\nStopped by Ctrl+C")

print("\nTest complete!")
