#!/usr/bin/env python3
"""
Test script to check Open WebUI configuration and available models
"""

import httpx
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://216.81.245.140:8080")
OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN")

if not OPENWEBUI_TOKEN:
    print("Error: OPENWEBUI_TOKEN not found in .env file")
    sys.exit(1)

print("\n" + "="*60)
print("Testing Open WebUI Connection")
print("="*60 + "\n")

# Test 1: Connection test
print("1. Testing connection to Open WebUI...")
try:
    with httpx.Client(timeout=10.0) as client:
        headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
        response = client.get(f"{OPENWEBUI_URL}/api/config", headers=headers)
        response.raise_for_status()
        config = response.json()
        print("✓ Connected successfully!")
        print(f"   Version: {config.get('version', 'N/A')}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    sys.exit(1)

# Test 2: Check different model endpoints
print("\n2. Checking different model endpoints...")

endpoints_to_try = [
    "/api/models",
    "/api/v1/models", 
    "/ollama/api/tags",
]

found_models = []
for endpoint in endpoints_to_try:
    print(f"\n   Trying: {OPENWEBUI_URL}{endpoint}")
    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
            response = client.get(f"{OPENWEBUI_URL}{endpoint}", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for models in different keys
                models = []
                if 'data' in data and data['data']:
                    models = data['data']
                    found_models = models
                    print(f"   ✓ Found {len(models)} models!")
                    for model in models[:5]:
                        print(f"      Model ID: {model.get('id', 'N/A')}")
                elif 'models' in data and data['models']:
                    models = data['models']
                    found_models = models
                    print(f"   ✓ Found {len(models)} models!")
                    for model in models[:5]:
                        print(f"      Model: {model.get('name', model.get('id', 'N/A'))}")
                else:
                    print(f"   ✗ Found 0 models")
                    
            else:
                print(f"   ✗ Status {response.status_code}")
                
    except Exception as e:
        print(f"   ✗ Error: {e}")

# Test 3: Get TotalGPT models directly
print("\n3. Fetching models from TotalGPT...")
totalgpt_token = "sk-dOwBzdjuw0OWIgYAyytZoA"

totalgpt_models = []
try:
    with httpx.Client(timeout=10.0) as client:
        headers = {"Authorization": f"Bearer {totalgpt_token}"}
        response = client.get("https://api.totalgpt.ai/v1/models", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            totalgpt_models = data.get('data', [])
            
            print(f"   ✓ TotalGPT API accessible!")
            print(f"   Available models ({len(totalgpt_models)} total):")
            
            # Buscar el modelo Qwen específico
            for model in totalgpt_models:
                model_id = model.get('id', '')
                if 'Qwen' in model_id or 'qwen' in model_id:
                    print(f"      - {model_id}")
        else:
            print(f"   ✗ TotalGPT API error: {response.status_code}")
            
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test configured model with all possible formats
print("\n4. Testing the exact Model ID you configured...")
configured_model = "Qwen-Qwen3-30B-A3B"

test_formats = [
    configured_model,
    f"totalgpt:{configured_model}",  
    f"totalgpt/{configured_model}",
    configured_model.lower(),
    f"openai/{configured_model}",
]

# También probar los modelos que encontramos en TotalGPT
for model in totalgpt_models:
    model_id = model.get('id', '')
    if 'Qwen' in model_id and 'A3B' in model_id:
        test_formats.insert(0, model_id)  # Prioritize exact match
        break

model_found = False
for test_model in test_formats:
    print(f"\n   Testing: {test_model}")
    try:
        with httpx.Client(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": test_model,
                "messages": [{"role": "user", "content": "Di solo 'ok'"}],
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
                print(f"\n   ★★★ MODEL NAME: {test_model} ★★★")
                model_found = True
                break
            else:
                print(f"   ✗ {response.status_code}: {response.text[:80]}")
                
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}")

# Test 5: Debug - check Open WebUI config
print("\n5. Checking Open WebUI configuration...")
try:
    with httpx.Client(timeout=10.0) as client:
        headers = {"Authorization": f"Bearer {OPENWEBUI_TOKEN}"}
        
        response = client.get(f"{OPENWEBUI_URL}/api/config", headers=headers)
        if response.status_code == 200:
            config = response.json()
            print(f"   Config keys: {list(config.keys())[:10]}")
            
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if model_found:
    print("\n✅ MODELO FUNCIONANDO!")
    print("Copia el nombre del modelo marcado con ★★★ arriba")
    print("y actualiza DEFAULT_MODEL en config.py")
elif found_models:
    print("\n⚠️ Se encontraron modelos en Open WebUI pero ninguno funcionó")
    print("Prueba uno de los modelos listados arriba manualmente")
else:
    print("\n❌ El modelo configurado no funciona.")
    print("\nPosibles soluciones:")
    print("\n1. VERIFICA LA CONFIGURACIÓN EN OPEN WEBUI:")
    print("   - URL: https://api.totalgpt.ai/v1 (CON /v1)")
    print("   - Model IDs: déjalo VACÍO (que detecte automáticamente)")
    print("   - Prefix ID: déjalo VACÍO")
    print("   - Save y espera 10 segundos")
    print("\n2. REINICIA OPEN WEBUI:")
    print("   docker restart open-webui")
    print("\n3. ALTERNATIVA DIRECTA:")
    print("   - Puedo configurar BioBot para usar TotalGPT directamente")
    print("   - Sin pasar por Open WebUI")
    print("   - Más simple y directo")
    print("   - ¿Quieres que haga esto?")

print("="*60)
