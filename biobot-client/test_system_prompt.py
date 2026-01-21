import os
import httpx

OPENWEBUI_URL = os.environ.get("OPENWEBUI_URL", "http://192.168.1.5:3000")
OPENWEBUI_TOKEN = os.environ.get("OPENWEBUI_TOKEN", "sk-39d983fbeff241a89a5400e7a944fe16")
MODEL = "llama3.2-vision:11b"
SYSTEM_PROMPT = "Eres un asistente visual breve y directo. Responde en español."

headers = {
    "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
    "Content-Type": "application/json"
}

chat_data = {
    "name": "Test System Prompt",
    "models": [MODEL],
    "params": {"system": SYSTEM_PROMPT}
}

payload = {"chat": chat_data}

url = f"{OPENWEBUI_URL}/api/v1/chats/new"

if __name__ == "__main__":
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            print("Response:", result)
            print("System prompt set:", result.get("params", {}))
    except Exception as e:
        print("Error:", e)
