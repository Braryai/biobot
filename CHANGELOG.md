# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-22

### Added
- System prompt configuration via voice command
- Real-time debug logs panel in dashboard
- Support for Doctor-Shotgun-L3.3-70B-Magnum-v4-SE model
- Dynamic model display in dashboard debug info
- Voice command: "set system prompt [text]"
- Dashboard visual feedback for system prompt status

### Changed
- System prompt now included in every API request (simple approach)
- Improved dashboard layout with dedicated debug panel
- Updated recommended model for better system prompt support
- Cleaned up voice commands panel (removed unimplemented features)
- All code comments translated to English
- Updated README with v1.1.0 features and troubleshooting

### Fixed
- System prompt not being applied to queries
- Dashboard not updating current model correctly
- Debug information not visible to user (now in dashboard)

### Removed
- Test scripts (test_system_prompt*.py)
- Development documentation files
- Unused voice command implementations

### Security
- Verified no hardcoded API keys in codebase
- Removed sensitive data from test files

## [1.0.0] - 2025-01-15

### Added
- Initial release of BioBot voice client
- Core functionality for hands-free datacenter assistance
- Push-to-talk recording with two modes (voice+screenshot, voice only)
- Open WebUI API integration with RAG support
- Speech-to-text via OpenAI Whisper, Groq, or Local Whisper
- Optional text-to-speech responses
- Conversation history management
- Smart screenshot capture (focused window or full screen)
- Configurable keyboard shortcuts
- Automatic cleanup of temporary files
- Integration with Open WebUI and Knowledge Base
- Support for vision models with screenshot analysis
- Professional documentation and setup guides
