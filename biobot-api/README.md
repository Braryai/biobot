# BioBot API Services

This directory is reserved for future API services and backend components.

## Planned Features

- REST API for mobile clients
- WebSocket server for real-time communication
- Authentication and authorization services
- Conversation history storage
- Multi-user support
- Analytics and logging

## Current Status

⚠️ **Not yet implemented** - Currently, the BioBot client connects directly to Open WebUI.

## Future Architecture

```
Mobile App / Smart Glasses
         ↓
    BioBot API (this directory)
         ↓
    Open WebUI (216.81.245.140:8080)
         ↓
    Knowledge Base
```

## Environment Variables

See `.env.example` for configuration template when API services are implemented.
