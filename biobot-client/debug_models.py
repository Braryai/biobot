#!/usr/bin/env python3
"""
Debug script - inspecting models configuration in Open WebUI
"""

import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://216.81.245.140:8080")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN")

def inspect_endpoint(url, method="GET", data=None):
    """Helper to inspect any endpoint"""
    try:
        with httpx.Client(timeout=15.0) as client:
            headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
            
            if method == "GET":
                response = client.get(url, headers=headers)
            elif method == "POST":
                headers["Content-Type"] = "application/json"
                response = client.post(url, headers=headers, json=data)
            
            print(f"\n{'='*60}")
            print(f"{method} {url}")
            print(f"Status: {response.status_code}")
            print(f"{'='*60}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(json.dumps(data, indent=2))
                except:
                    print(response.text[:500])
            else:
                print(f"Error: {response.text[:300]}")
                
            return response
    except Exception as e:
        print(f"Exception: {e}")
        return None

print("\n" + "="*60)
print("DEBUGGING OPEN WEBUI MODELS")
print("="*60)

# 1. Get base models
print("\n### 1. GET BASE MODELS ###")
inspect_endpoint(f"{OPENWEBUI_URL}/api/v1/models/base")

# 2. Get all models list
print("\n### 2. GET MODELS LIST ###")
inspect_endpoint(f"{OPENWEBUI_URL}/api/v1/models/list")

# 3. Get models from /api/models (standard endpoint)
print("\n### 3. GET /api/models ###")
inspect_endpoint(f"{OPENWEBUI_URL}/api/models")

# 4. Get connections config
print("\n### 4. GET CONNECTIONS CONFIG ###")
inspect_endpoint(f"{OPENWEBUI_URL}/api/v1/configs/connections")

# 5. Get OpenAI config
print("\n### 5. GET OPENAI CONFIG ###")
inspect_endpoint(f"{OPENWEBUI_URL}/openai/config")

# 6. Get OpenAI models
print("\n### 6. GET OPENAI MODELS ###")
inspect_endpoint(f"{OPENWEBUI_URL}/openai/models")

# 7. Get models config
print("\n### 7. GET MODELS CONFIG ###")
inspect_endpoint(f"{OPENWEBUI_URL}/api/v1/configs/models")

# 8. Try chat completion with discovered model
print("\n### 8. TRY CHAT COMPLETION ###")
print("Testing different model formats...")

test_models = [
    "Qwen-Qwen3-30B-A3B",
    "gpt-3.5-turbo",  # Sometimes custom endpoints mimic OpenAI names
    "totalgpt/Qwen-Qwen3-30B-A3B",
]

for model_name in test_models:
    print(f"\nTrying: {model_name}")
    inspect_endpoint(
        f"{OPENWEBUI_URL}/api/chat/completions",
        method="POST",
        data={
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False
        }
    )

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("\n¿Encontraste algún modelo en las respuestas arriba?")
print("Si sí, úsalo en config.py como DEFAULT_MODEL")
print("="*60)
