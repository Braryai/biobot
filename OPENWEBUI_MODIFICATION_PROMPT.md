# Open WebUI Modification - Image Handling & Live Streaming for External API Clients

## CONTEXT

I have a voice-controlled datacenter assistant called **BioBot** that uses Open WebUI as the backend. BioBot is an external Python client that:

1. Captures audio from technicians wearing smart glasses
2. Transcribes speech using local Whisper
3. Optionally captures screenshots (base64 PNG images, ~500KB each)
4. Sends queries to Open WebUI via API at your configured server
5. Uses two models:
   - `Llama-3.2-11B-Vision-Instruct` (for queries WITH screenshots)
   - `Qwen-Qwen3-30B-A3B` (for text-only queries)

## CURRENT PROBLEM

When BioBot sends messages with screenshots to Open WebUI:

1. **Images in base64 format are TOO LARGE** (~500KB per image) to store directly in chat messages DB
2. **Frontend doesn't display images** sent via API - only shows text
3. **No auto-refresh** - frontend doesn't update when new API messages arrive
4. **No streaming responses** - responses come all at once after processing completes

## CURRENT IMPLEMENTATION (BioBot Client)

```python
# BioBot creates chat via API
POST /api/v1/chats/new
{
  "chat": {
    "name": "BioBot - 2026-01-08 10:30",
    "models": ["Llama-3.2-11B-Vision-Instruct"]
  }
}
# Returns: {"id": "abc123..."}

# BioBot gets AI response (with image processing)
POST /api/chat/completions
{
  "model": "Llama-3.2-11B-Vision-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this screenshot?"},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,iVBORw0KGgoA..."}
        }
      ]
    }
  ],
  "stream": false
}

# BioBot saves to frontend (currently TEXT ONLY - no images)
GET /api/v1/chats/{chat_id}
POST /api/v1/chats/{chat_id}
{
  "chat": {
    "messages": [
      {"role": "user", "content": "📸 [Screenshot included] What do you see?"},
      {"role": "assistant", "content": "I see a VS Code editor..."}
    ]
  }
}
```

## DESIRED SOLUTION

Modify Open WebUI to handle images from external API clients efficiently:

### 1. IMAGE STORAGE
- **Store images as files** instead of base64 in DB
- Create endpoint: `POST /api/v1/files/upload` that:
  - Accepts base64 image data
  - Saves to disk/storage (e.g., `/data/uploads/{chat_id}/{timestamp}.png`)
  - Returns file reference/URL
  - Cleans up old files after X days

### 2. MESSAGE FORMAT ENHANCEMENT
- Extend chat message schema to support image references:
```json
{
  "role": "user",
  "content": "What do you see in this screenshot?",
  "attachments": [
    {
      "type": "image",
      "url": "/uploads/{chat_id}/20260108_103045.png",
      "thumbnail": "/uploads/{chat_id}/20260108_103045_thumb.png"
    }
  ]
}
```

### 3. FRONTEND AUTO-REFRESH
- Add WebSocket/SSE support for chat updates
- When external API adds messages via `POST /api/v1/chats/{chat_id}`:
  - Broadcast update to connected WebSocket clients viewing that chat
  - Frontend automatically displays new messages without manual refresh

### 4. STREAMING RESPONSES
- Enable streaming for API clients:
```python
POST /api/chat/completions
{
  "model": "Llama-3.2-11B-Vision-Instruct",
  "messages": [...],
  "stream": true,  # Enable streaming
  "chat_id": "abc123"  # Link to frontend chat
}
```
- Stream response to both:
  - API client (current behavior)
  - Frontend WebSocket (new: live updates in UI)

## TECHNICAL REQUIREMENTS

### Backend Changes Needed:

1. **File Upload Endpoint** (`backend/apps/webui/routers/files.py` or similar)
   - Accept base64 images from API
   - Generate thumbnails for performance
   - Return file references
   - Implement cleanup jobs

2. **Message Schema Update** (`backend/apps/webui/models/chats.py`)
   - Add `attachments` field to Message model
   - Support image references in messages

3. **WebSocket/SSE for Chat Updates** (`backend/apps/socket/main.py`)
   - Emit events when chat is updated via API
   - Include room/channel per chat_id
   - Send new messages to connected clients

4. **Streaming Enhancement** (`backend/apps/openai/main.py`)
   - When `stream=true` + `chat_id` provided:
     - Stream to API client as usual
     - Also emit chunks to WebSocket for frontend
     - Save complete message after streaming ends

### Frontend Changes Needed:

1. **Image Display Component** (`src/lib/components/chat/Messages.svelte`)
   - Render images from attachment URLs
   - Show thumbnails with lightbox for full view
   - Handle lazy loading

2. **WebSocket Client** (`src/lib/apis/streaming.js`)
   - Connect to WebSocket on chat page load
   - Subscribe to chat_id room
   - Update UI when new messages arrive
   - Handle streaming message updates

3. **Auto-Refresh Logic** (`src/routes/c/[id]/+page.svelte`)
   - Listen for WebSocket events
   - Append new messages to chat history
   - Scroll to bottom on new message
   - Show "New message" indicator if scrolled up

## CURRENT OPEN WEBUI ARCHITECTURE (Based on v0.6.43)

- **Backend**: Python FastAPI
- **Frontend**: SvelteKit
- **Database**: SQLite (default) or PostgreSQL
- **File Storage**: Local filesystem `/data/uploads/`
- **Real-time**: Currently limited, needs WebSocket enhancement

## SUCCESS CRITERIA

After implementation, BioBot should:

1. ✅ Send image via API → Image stored as file, not base64 in DB
2. ✅ Open WebUI frontend displays image in chat history
3. ✅ When BioBot sends message → Frontend auto-updates (no manual refresh)
4. ✅ Streaming responses appear live in frontend (character by character)
5. ✅ Images persist and are viewable in chat history
6. ✅ Performance: No DB bloat, fast page loads

## EXAMPLE WORKFLOW AFTER IMPLEMENTATION

```python
# 1. BioBot uploads image
POST /api/v1/files/upload
{
  "chat_id": "abc123",
  "image": "data:image/png;base64,iVBORw0KGgoA..."
}
# Response: {"file_id": "xyz789", "url": "/uploads/abc123/20260108_103045.png"}

# 2. BioBot sends message with image reference
POST /api/chat/completions
{
  "model": "Llama-3.2-11B-Vision-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ],
      "attachments": [{"type": "image", "file_id": "xyz789"}]
    }
  ],
  "stream": true,
  "chat_id": "abc123"
}

# 3. Open WebUI:
#    - Processes image with vision model
#    - Streams response to API client AND WebSocket
#    - Saves message with image reference to DB
#    - Frontend shows image + streaming response live

# 4. Technician opens Open WebUI frontend:
#    - Sees full chat history
#    - Images displayed inline
#    - New messages appear automatically
#    - Responses stream live
```

## FILES TO MODIFY (Estimated)

### Backend:
- `backend/apps/webui/routers/files.py` (new or extend existing)
- `backend/apps/webui/models/chats.py`
- `backend/apps/openai/main.py`
- `backend/apps/socket/main.py` (create if doesn't exist)
- `backend/config.py` (add file storage settings)

### Frontend:
- `src/lib/components/chat/Messages.svelte`
- `src/lib/components/chat/MessageInput.svelte`
- `src/lib/apis/chats.js`
- `src/lib/apis/streaming.js` (create if doesn't exist)
- `src/routes/c/[id]/+page.svelte`

## PRIORITY ORDER

1. **HIGH**: Image file storage (fixes DB bloat issue)
2. **HIGH**: Display images in frontend
3. **MEDIUM**: Auto-refresh via WebSocket
4. **MEDIUM**: Streaming responses to frontend
5. **LOW**: Thumbnail generation, cleanup jobs

## ADDITIONAL CONTEXT

- Open WebUI version: 0.6.43
- Deployment: Single server configured in config.py
- Users: Datacenter technicians using smart glasses
- Use case: Hands-free troubleshooting with visual context
- Critical: Images must be preserved for audit trail
- Performance: Technicians need responses in < 5 seconds

## QUESTION FOR COPILOT

Please help me implement this image handling and live streaming functionality in Open WebUI. Start with the image file storage system, then guide me through the other components. Show me:

1. Where to add the file upload endpoint
2. How to modify the message schema
3. How to integrate WebSocket for auto-refresh
4. How to enable streaming to frontend

Provide code examples for each component, considering the existing Open WebUI architecture.
