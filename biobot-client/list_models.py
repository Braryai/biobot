#!/usr/bin/env python3
"""
List all available models in Open WebUI
"""

import httpx
import os
from dotenv import load_dotenv

load_dotenv()

OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://YOUR_SERVER_URL")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN")

print("\n" + "="*60)
print("LISTING ALL AVAILABLE MODELS")
print("="*60 + "\n")

try:
    with httpx.Client(timeout=10.0) as client:
        headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
        response = client.get(f"{OPENWEBUI_URL}/api/models", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            
            print(f"Found {len(models)} models:\n")
            
            # Filter vision/multimodal models
            vision_models = []
            
            for i, model in enumerate(models, 1):
                model_id = model.get('id', 'N/A')
                print(f"{i}. {model_id}")
                
                # Check if it's a vision model
                if any(keyword in model_id.lower() for keyword in ['vision', 'llama-3.2', 'llava', 'gpt-4', 'gemini']):
                    vision_models.append(model_id)
            
            if vision_models:
                print("\n" + "="*60)
                print("VISION/MULTIMODAL MODELS (support images):")
                print("="*60)
                for model_id in vision_models:
                    print(f"  ✓ {model_id}")
                    
                print("\n" + "="*60)
                print("RECOMMENDED MODEL:")
                print("="*60)
                print(f"\nUse this in config.py:\n")
                print(f'DEFAULT_MODEL = "{vision_models[0]}"')
            else:
                print("\n⚠ No vision models found!")
                print("Using non-vision model will work for text-only queries")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
