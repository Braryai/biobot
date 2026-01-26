# BioBot v1.1.0 Release Notes

## Overview
Version 1.1.0 introduces system prompt configuration via voice commands, an improved dashboard with debug logs, and better model support.

## New Features

### System Prompt Configuration
- Configure AI behavior via voice: "set system prompt [text]"
- System prompt applies to all subsequent messages
- Visual confirmation in dashboard debug panel
- Simple implementation: included in every API request

### Debug Logs Panel
- Real-time debug information visible in dashboard
- Color-coded log levels (info, warning, error, success)
- Shows system prompt status and application
- Last 10 logs displayed with timestamps

### Improved Model Support
- Recommended model: Doctor-Shotgun-L3.3-70B-Magnum-v4-SE
- Better system prompt adherence
- Dynamic model display in dashboard

## Improvements

### Dashboard
- Cleaner voice commands panel (only working commands shown)
- Debug logs panel for troubleshooting
- Current model updates dynamically
- Better visual feedback for all operations

### Code Quality
- All comments translated to English
- No hardcoded API keys or sensitive data
- Removed test files and dev documentation
- Clean production-ready codebase

### Documentation
- Updated README with v1.1.0 features
- Added troubleshooting for system prompts
- Comprehensive .cursorrules for development
- Updated CHANGELOG with detailed changes

## Breaking Changes
None. Fully backward compatible with v1.0.0 configurations.

## Migration Guide
If upgrading from v1.0.0:

1. Pull latest code
2. Update config.py if needed (model recommendation changed)
3. Run `pip install -r requirements.txt` (added rich for dashboard)
4. No other changes required

## Bug Fixes
- System prompt now correctly applied to all queries
- Dashboard debug info shows current model
- Debug information now visible to users (in dashboard, not just terminal)

## Files Changed
- `biobot_voice.py`: System prompt in requests, dashboard integration
- `dashboard.py`: Added debug panel, cleaned voice commands
- `config.py`: Updated recommended model
- `requirements.txt`: Added rich library
- `README.md`: Updated features and troubleshooting
- `.cursorrules`: Created with current implementation details
- `CHANGELOG.md`: Added v1.1.0 entry

## Files Removed
- `test_system_prompt*.py`: Test files (4 files)
- `*_SISTEMA_PROMPT.md`: Development documentation (3 files)

## Known Issues
None

## Future Roadmap
- Additional voice commands
- Multi-platform support
- Enhanced screenshot tools
- Better error recovery

## Contributors
- Development team

## Support
For issues and questions, visit the GitHub repository.

---

Released: January 22, 2026
