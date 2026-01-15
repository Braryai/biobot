#!/usr/bin/env python3
"""Quick script to check available models in Open WebUI"""

import httpx
from config import OPENWEBUI_URL, OPENWEBUI_TOKEN

headers = {
    "Authorization": f"Bearer {OPENWEBUI_TOKEN}"
}

print(f"Checking models at: {OPENWEBUI_URL}")
print("="*60)

try:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{OPENWEBUI_URL}/api/models", headers=headers)
        response.raise_for_status()
        models = response.json()
        
        print(f"\n✓ Found {len(models.get('data', []))} models:\n")
        
        for model in models.get('data', []):
            model_id = model.get('id', 'unknown')
            name = model.get('name', model_id)
            print(f"  • {model_id}")
            if name != model_id:
                print(f"    Name: {name}")
        
        print("\n" + "="*60)
        print("\nUpdate config.py with the correct model IDs")
        
except Exception as e:
    print(f"❌ Error: {e}")
