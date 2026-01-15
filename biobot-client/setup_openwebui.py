#!/usr/bin/env python3
"""
Script para configurar TotalGPT endpoint en Open WebUI via API
"""

import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://216.81.245.140:8080")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN")
TOTALGPT_API_KEY = "sk-dOwBzdjuw0OWIgYAyytZoA"

if not OPENWEBUI_TOKEN:
    print("Error: OPENWEBUI_TOKEN not found in .env file")
    exit(1)

print("\n" + "="*60)
print("Open WebUI - TotalGPT Endpoint Setup")
print("="*60 + "\n")

# Paso 1: Listar conexiones existentes
print("1. Checking existing connections...")
try:
    with httpx.Client(timeout=10.0) as client:
        headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
        
        # Intentar obtener configuraciones de OpenAI
        response = client.get(f"{OPENWEBUI_URL}/api/configs", headers=headers)
        if response.status_code == 200:
            configs = response.json()
            print(f"   Current configs: {json.dumps(configs, indent=2)[:500]}")
        
        # Intentar obtener connections
        response = client.get(f"{OPENWEBUI_URL}/api/connections", headers=headers)
        if response.status_code == 200:
            connections = response.json()
            print(f"   Current connections: {json.dumps(connections, indent=2)[:500]}")
        
except Exception as e:
    print(f"   Error: {e}")

# Paso 2: Agregar TotalGPT como custom endpoint
print("\n2. Adding TotalGPT as OpenAI-compatible endpoint...")
try:
    with httpx.Client(timeout=10.0) as client:
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Formato para agregar OpenAI API compatible endpoint
        payload = {
            "name": "TotalGPT",
            "url": "https://api.totalgpt.ai/v1",
            "api_key": TOTALGPT_API_KEY,
            "type": "openai"
        }
        
        # Intentar varios endpoints posibles
        endpoints_to_try = [
            "/api/connections",
            "/api/openai/config",
            "/api/configs/openai"
        ]
        
        for endpoint in endpoints_to_try:
            print(f"\n   Trying: {endpoint}")
            try:
                response = client.post(
                    f"{OPENWEBUI_URL}{endpoint}",
                    json=payload,
                    headers=headers
                )
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text[:300]}")
                
                if response.status_code == 200:
                    print(f"   ✓ SUCCESS!")
                    break
            except Exception as e:
                print(f"   Error: {e}")
                
except Exception as e:
    print(f"   Error: {e}")

# Paso 3: Probar si ahora aparecen modelos
print("\n3. Checking if models now appear...")
try:
    with httpx.Client(timeout=10.0) as client:
        headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
        response = client.get(f"{OPENWEBUI_URL}/api/models", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {len(data.get('data', []))} models")
            
            for model in data.get('data', [])[:5]:
                print(f"   - {model.get('id', 'N/A')}")
                
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("\n1. Si esto no funcionó, debes agregar el endpoint manualmente:")
print("   a. Ve a: http://216.81.245.140:8080")
print("   b. Haz login")
print("   c. Settings → Connections → Add Connection")
print("   d. Tipo: OpenAI API")
print("   e. Name: TotalGPT")
print("   f. URL: https://api.totalgpt.ai/v1")
print("   g. API Key: sk-dOwBzdjuw0OWIgYAyytZoA")
print("   h. Guardar y refrescar")
print("\n2. Una vez agregado, los modelos aparecerán con prefijo")
print("   Ejemplo: 'totalgpt/TheDrummer-Anubis-70B-v1-FP8-Dynamic'")
print("\n3. O puedes llamar directamente sin que aparezca en la lista")
print("   usando el model ID completo desde TotalGPT")
print("="*60)
