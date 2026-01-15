#!/usr/bin/env python3
"""
Test chat creation and continuation in Open WebUI
"""

import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://216.81.245.140:8080")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN")

print("\n" + "="*60)
print("TESTING CHAT CREATION & CONTINUATION")
print("="*60)

# Step 1: Create a new chat
print("\n1. Creating new chat...")
try:
    with httpx.Client(timeout=15.0) as client:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Create chat payload
        payload = {
            "chat": {
                "name": "BioBot Test Chat",
                "models": ["Qwen-Qwen3-30B-A3B"]
            }
        }
        
        response = client.post(
            f"{OPENWEBUI_URL}/api/v1/chats/new",
            json=payload,
            headers=headers
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)[:500]}")
        
        if response.status_code == 200:
            chat_data = response.json()
            chat_id = chat_data.get('id')
            print(f"\n   ✓ Chat created! ID: {chat_id}")
            
            # Step 2: Send first message via chat completions
            print(f"\n2. Sending first message...")
            completion_payload = {
                "model": "Qwen-Qwen3-30B-A3B",
                "messages": [
                    {"role": "user", "content": "Hola, esto es una prueba"}
                ],
                "stream": False,
                "chat_id": chat_id  # Link to chat
            }
            
            response = client.post(
                f"{OPENWEBUI_URL}/api/chat/completions",
                json=completion_payload,
                headers=headers
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Response received")
                print(f"   Content: {result.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}")
            else:
                print(f"   Error: {response.text}")
            
            # Step 3: Send follow-up message
            print(f"\n3. Sending follow-up message...")
            continuation_payload = {
                "model": "Qwen-Qwen3-30B-A3B",
                "messages": [
                    {"role": "user", "content": "Hola, esto es una prueba"},
                    {"role": "assistant", "content": result.get('choices', [{}])[0].get('message', {}).get('content', '')},
                    {"role": "user", "content": "¿Recuerdas qué te dije antes?"}
                ],
                "stream": False,
                "chat_id": chat_id
            }
            
            response = client.post(
                f"{OPENWEBUI_URL}/api/chat/completions",
                json=continuation_payload,
                headers=headers
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result2 = response.json()
                print(f"   ✓ Response received")
                print(f"   Content: {result2.get('choices', [{}])[0].get('message', {}).get('content', '')[:150]}")
            else:
                print(f"   Error: {response.text}")
            
            # Step 4: Verify chat exists
            print(f"\n4. Verifying chat exists in Open WebUI...")
            response = client.get(
                f"{OPENWEBUI_URL}/api/v1/chats/",
                headers=headers
            )
            
            if response.status_code == 200:
                chats = response.json()
                print(f"   Found {len(chats)} chats total")
                
                # Find our chat
                our_chat = [c for c in chats if c.get('id') == chat_id]
                if our_chat:
                    print(f"   ✓ Our chat exists!")
                    print(f"   Title: {our_chat[0].get('title', 'N/A')}")
                else:
                    print(f"   ✗ Chat not found in list")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("Go check: http://216.81.245.140:8080")
print("The chat should appear in Workspace → Chats")
print("="*60)
