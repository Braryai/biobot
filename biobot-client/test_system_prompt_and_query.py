#!/usr/bin/env python3
"""
Test script for System Prompt workflow in Open WebUI
Strategy: Recreate chat with system prompt after first message
"""

import os
import httpx
from datetime import datetime

# Configuration
OPENWEBUI_URL = os.environ.get("OPENWEBUI_URL", "http://192.168.1.5:3000")
OPENWEBUI_TOKEN = os.environ.get("OPENWEBUI_TOKEN", "sk-39d983fbeff241a89a5400e7a944fe16")
MODEL = "Llama-3.2-11B-Vision-Instruct"
SYSTEM_PROMPT = "Eres un asistente visual breve y directo. Responde en español de forma concisa."

headers = {
    "Authorization": f"Bearer {OPENWEBUI_TOKEN}",
    "Content-Type": "application/json"
}


def create_chat(model: str, name: str = None, system_prompt: str = None) -> str:
    """Create a new chat with optional system prompt."""
    if name is None:
        name = f"Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    chat_data = {
        "title": name,
        "models": [model]
    }
    
    # Add system prompt if provided
    if system_prompt:
        chat_data["params"] = {"system": system_prompt}
        print(f"   📝 System prompt incluido: '{system_prompt[:50]}...'")
    
    payload = {"chat": chat_data}
    url = f"{OPENWEBUI_URL}/api/v1/chats/new"
    
    with httpx.Client(timeout=10.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        chat_id = result["id"]
        print(f"✓ Chat creado: {chat_id}")
        return chat_id


def query_model(question: str, model: str, conversation_history: list = None) -> str:
    """Query the model and return the response (does NOT save to chat)."""
    messages = conversation_history or []
    messages.append({"role": "user", "content": question})
    
    payload = {
        "messages": messages,
        "model": model,
        "stream": False
    }
    
    url = f"{OPENWEBUI_URL}/api/chat/completions"
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        assistant_response = result["choices"][0]["message"]["content"]
        print(f"✓ Respuesta recibida: {assistant_response[:80]}...")
        return assistant_response


def save_message_to_chat(chat_id: str, user_message: str, assistant_message: str) -> bool:
    """Save user and assistant messages to the chat."""
    get_url = f"{OPENWEBUI_URL}/api/v1/chats/{chat_id}"
    
    with httpx.Client(timeout=30.0) as client:
        # Get existing messages
        get_response = client.get(get_url, headers=headers)
        get_response.raise_for_status()
        chat_data = get_response.json()
        
        # Append new messages
        existing_messages = chat_data["chat"].get("messages", [])
        new_messages = existing_messages + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        
        # Update chat
        chat_data["chat"]["messages"] = new_messages
        update_url = f"{OPENWEBUI_URL}/api/v1/chats/{chat_id}"
        update_response = client.post(update_url, json={"chat": chat_data["chat"]}, headers=headers)
        update_response.raise_for_status()
        print(f"✓ Mensajes guardados en el chat")
        return True


def get_chat_messages(chat_id: str) -> list:
    """Get all messages from a chat."""
    get_url = f"{OPENWEBUI_URL}/api/v1/chats/{chat_id}"
    
    with httpx.Client(timeout=10.0) as client:
        response = client.get(get_url, headers=headers)
        response.raise_for_status()
        chat_data = response.json()
        return chat_data["chat"].get("messages", [])


def recreate_chat_with_system_prompt(old_chat_id: str, system_prompt: str, model: str) -> str:
    """Recreate chat with system prompt, preserving message history."""
    print(f"\n🔄 Recreando chat con system prompt...")
    
    # Get existing messages
    messages = get_chat_messages(old_chat_id)
    print(f"   📜 Recuperando {len(messages)} mensajes del chat anterior")
    
    # Create new chat with system prompt
    new_chat_id = create_chat(model, "Chat con System Prompt", system_prompt)
    
    # Copy messages to new chat (in bulk)
    if messages:
        get_url = f"{OPENWEBUI_URL}/api/v1/chats/{new_chat_id}"
        
        with httpx.Client(timeout=30.0) as client:
            get_response = client.get(get_url, headers=headers)
            get_response.raise_for_status()
            chat_data = get_response.json()
            
            # Set messages
            chat_data["chat"]["messages"] = messages
            
            # Update chat
            update_url = f"{OPENWEBUI_URL}/api/v1/chats/{new_chat_id}"
            update_response = client.post(update_url, json={"chat": chat_data["chat"]}, headers=headers)
            update_response.raise_for_status()
            print(f"✓ Mensajes copiados al nuevo chat")
    
    # Optionally delete old chat
    try:
        delete_url = f"{OPENWEBUI_URL}/api/v1/chats/{old_chat_id}"
        with httpx.Client(timeout=10.0) as client:
            client.delete(delete_url, headers=headers)
            print(f"✓ Chat anterior eliminado")
    except:
        print(f"⚠️  No se pudo eliminar el chat anterior")
    
    return new_chat_id


def verify_chat(chat_id: str):
    """Verify chat contents and system prompt."""
    get_url = f"{OPENWEBUI_URL}/api/v1/chats/{chat_id}"
    
    with httpx.Client(timeout=10.0) as client:
        response = client.get(get_url, headers=headers)
        response.raise_for_status()
        chat_data = response.json()
        
        messages = chat_data["chat"].get("messages", [])
        system = chat_data["chat"].get("params", {}).get("system", "N/A")
        
        print("\n" + "="*60)
        print("VERIFICACIÓN DEL CHAT")
        print("="*60)
        print(f"Chat ID: {chat_id}")
        print(f"Nombre: {chat_data['chat'].get('title', 'N/A')}")
        print(f"Modelo: {chat_data['chat'].get('models', ['N/A'])[0]}")
        print(f"Mensajes: {len(messages)}")
        print(f"System Prompt: {system[:100] if system != 'N/A' else 'N/A'}...")
        print("="*60)
        
        if messages:
            print("\nMENSAJES EN EL CHAT:")
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                print(f"\n{i}. [{role.upper()}]")
                print(f"   {content[:150]}...")
        
        return chat_data


def main():
    """Test the complete workflow."""
    print("\n" + "="*60)
    print("TEST: System Prompt Workflow (Recreate Strategy)")
    print("="*60 + "\n")
    
    try:
        # PASO 1: Crear chat SIN system prompt
        print("PASO 1: Creando chat inicial (sin system prompt)...")
        chat_id = create_chat(MODEL, "Test Initial Chat")
        
        # PASO 2: Enviar primer mensaje
        print("\nPASO 2: Enviando primer mensaje...")
        question1 = "¿Cuál es la capital de Francia?"
        response1 = query_model(question1, MODEL)
        
        # PASO 3: Guardar mensaje en el chat
        print("\nPASO 3: Guardando mensaje en el chat...")
        save_message_to_chat(chat_id, question1, response1)
        
        # PASO 4: Verificar chat inicial
        print("\nPASO 4: Verificando chat inicial...")
        verify_chat(chat_id)
        
        # PASO 5: RECREAR chat con system prompt
        print("\n" + "="*60)
        print("PASO 5: Aplicando system prompt (recreando chat)...")
        print("="*60)
        new_chat_id = recreate_chat_with_system_prompt(chat_id, SYSTEM_PROMPT, MODEL)
        chat_id = new_chat_id  # Use new chat ID from now on
        
        # PASO 6: Verificar nuevo chat
        print("\nPASO 6: Verificando nuevo chat con system prompt...")
        verify_chat(chat_id)
        
        # PASO 7: Enviar segundo mensaje (con system prompt activo)
        print("\n" + "="*60)
        print("PASO 7: Probando con segundo mensaje (con system prompt)...")
        print("="*60 + "\n")
        
        # Obtener historial para contexto
        conversation_history = get_chat_messages(chat_id)
        
        question2 = "¿Y cuál es su población aproximada?"
        response2 = query_model(question2, MODEL, conversation_history)
        save_message_to_chat(chat_id, question2, response2)
        
        # Verificación final
        print("\n" + "="*60)
        print("VERIFICACIÓN FINAL")
        print("="*60)
        verify_chat(chat_id)
        
        print("\n✅ TEST COMPLETADO EXITOSAMENTE")
        print(f"\nPuedes ver el chat en: {OPENWEBUI_URL}/c/{chat_id}")
        
    except httpx.HTTPStatusError as e:
        print(f"\n❌ Error HTTP {e.response.status_code}")
        print(f"URL: {e.request.url}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
