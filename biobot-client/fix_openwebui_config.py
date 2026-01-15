#!/usr/bin/env python3
"""
Configure TotalGPT endpoint in Open WebUI via API
"""

import httpx
import json
from config import OPENWEBUI_URL, OPENWEBUI_TOKEN

TOTALGPT_API_KEY = "sk-dOwBzdjuw0OWIgYAyytZoA"

print("\n" + "="*60)
print("CONFIGURING TOTALGPT IN OPEN WEBUI")
print(f"Server: {OPENWEBUI_URL}")
print("="*60)

# Set OpenAI config to use TotalGPT
print("\n1. Updating OpenAI config to use TotalGPT...")
try:
    with httpx.Client(timeout=15.0) as client:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "ENABLE_OPENAI_API": True,
            "OPENAI_API_BASE_URLS": ["https://api.totalgpt.ai/v1"],
            "OPENAI_API_KEYS": [TOTALGPT_API_KEY],
            "OPENAI_API_CONFIGS": {"0": {}}
        }
        
        response = client.post(
            f"{OPENWEBUI_URL}/openai/config/update",
            json=payload,
            headers=headers
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
        if response.status_code == 200:
            print("   ✓ OpenAI config updated!")
        else:
            print(f"   ✗ Failed: {response.text}")
            
except Exception as e:
    print(f"   ✗ Error: {e}")

# Wait a bit for Open WebUI to sync
print("\n2. Waiting for Open WebUI to sync...")
import time
time.sleep(3)

# Verify models are now available
print("\n3. Checking if models appear now...")
try:
    with httpx.Client(timeout=10.0) as client:
        headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
        
        response = client.get(f"{OPENWEBUI_URL}/api/models", headers=headers)
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            
            if models:
                print(f"   ✓ Found {len(models)} models!")
                for model in models[:10]:
                    print(f"      - {model.get('id', 'N/A')}")
            else:
                print("   ✗ Still 0 models. Trying OpenAI endpoint...")
                
                response = client.get(f"{OPENWEBUI_URL}/openai/models", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('data', [])
                    print(f"   Found {len(models)} models in OpenAI endpoint:")
                    for model in models[:10]:
                        print(f"      - {model.get('id', 'N/A')}")
        
except Exception as e:
    print(f"   Error: {e}")

# Test chat completion
print("\n4. Testing chat completion...")
try:
    with httpx.Client(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Qwen-Qwen3-30B-A3B",
            "messages": [{"role": "user", "content": "Say OK"}],
            "stream": False
        }
        
        response = client.post(
            f"{OPENWEBUI_URL}/api/chat/completions",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"   ✓✓✓ SUCCESS!")
            print(f"   Response: {content}")
        else:
            print(f"   ✗ Status {response.status_code}: {response.text[:200]}")
            
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
print("\nSi funcionó, actualiza config.py:")
print("DEFAULT_MODEL = 'Qwen-Qwen3-30B-A3B'")
print("="*60)
