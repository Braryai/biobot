# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Voice-controlled AI assistant for datacenter operations
- Push-to-talk recording with two modes (voice+screenshot, voice only)
- Open WebUI API integration with RAG support
- Speech-to-text via OpenAI Whisper or Groq
- Optional text-to-speech responses
- Conversation history management
- Smart screenshot capture (focused window or full screen)
- Configurable keyboard shortcuts
- Automatic cleanup of temporary files

### Security
- Removed leaked API keys and server URLs from codebase
- Added comprehensive .gitignore for sensitive files
- Updated all documentation with placeholder values

## [1.0.0] - 2025-01-15

### Added
- Initial release of BioBot voice client
- Core functionality for hands-free datacenter assistance
- Integration with Open WebUI and Knowledge Base
- Support for vision models with screenshot analysis
- Multiple STT providers (OpenAI, Groq, Local Whisper)
- Professional documentation and setup guides
