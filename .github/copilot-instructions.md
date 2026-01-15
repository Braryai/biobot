GLOBAL INSTRUCTIONS FOR THIS PROJECT:

CODE GENERATION RULES:

Generate ONLY executable code files, no explanatory text files
NO markdown documentation files unless explicitly requested
NO README updates unless explicitly requested
NO emoji in code or comments
ALL code comments must be in English
Follow PEP 8 style guide for Python
Use type hints for all function parameters and return values
Include docstrings only for public functions and classes
CONTEXT AWARENESS:

This is BioBot, a datacenter assistant using Open WebUI with RAG
Always remember we are working with: Open WebUI API, Ollama, llama3.1:8b, Knowledge Base for datacenter documentation
Current tech stack: Python 3.11+, FastAPI (if needed), httpx for HTTP calls, OpenAI Whisper for STT
Target platform: macOS
Deployment: Local development, single server
SECURITY BEST PRACTICES:

Never hardcode API keys, tokens, or credentials in code
Always use environment variables for sensitive data
Validate all user inputs before processing
Sanitize file paths to prevent directory traversal
Use HTTPS for external API calls when possible
Implement proper error handling without exposing sensitive information in error messages
Use secure random generation for tokens or IDs if needed
Implement rate limiting considerations for API calls
Log security-relevant events but never log credentials
CODE QUALITY STANDARDS:

Write defensive code with proper error handling
Use try-except blocks for external API calls and file operations
Validate environment variables on startup
Implement timeouts for all HTTP requests
Use context managers for file and network operations
Avoid global mutable state
Keep functions small and single-purpose
Use meaningful variable and function names
Prefer explicit over implicit code
ARCHITECTURE PRINCIPLES:

Separation of concerns between modules
Keep configuration separate from logic
Make code testable (avoid tight coupling)
Use dependency injection where appropriate
Follow SOLID principles
Prefer composition over inheritance
ERROR HANDLING:

Catch specific exceptions, not bare except
Provide meaningful error messages for debugging
Fail gracefully with user-friendly messages
Log errors with sufficient context
Never expose stack traces to end users in production
PERFORMANCE CONSIDERATIONS:

Minimize API calls where possible
Implement caching for repeated queries if applicable
Use async/await for I/O operations when beneficial
Close resources properly (files, connections, streams)
Avoid blocking operations in main thread
TESTING EXPECTATIONS:

Code should be structured to allow unit testing
Include basic input validation
Consider edge cases in implementation
Make external dependencies mockable
OUTPUT FORMAT:

When I ask for code modifications, provide ONLY the code
When I ask for file structure, provide ONLY the structure
When I ask for explanations, be concise and technical
Do not provide unsolicited advice or alternatives unless I ask for options
Do not create example files or placeholder content unless requested
WHEN MAKING CHANGES:

Modify existing files in place when possible
Only create new files when necessary for the architecture
Preserve existing functionality unless explicitly told to remove it
Maintain backward compatibility unless breaking changes are required
Update imports and dependencies as needed
FORBIDDEN ACTIONS (unless explicitly requested):

Do not create TODO.md, NOTES.md, or similar documentation files
Do not create example or template files
Do not add verbose comments explaining obvious code
Do not add ASCII art or decorative elements
Do not create backup or .old files
Do not add print statements for debugging (use logging module)
RESPONSE FORMAT:
When providing code:

Show the file path
Show the complete file content or the specific changes
If multiple files, clearly separate them
No explanatory prose before or after unless I ask "explain"
When I say "explain", then provide technical explanation with:

What changed and why
Security implications if any
Performance considerations if relevant
Dependencies added or removed
ASSUME I KNOW:

Python programming
REST APIs and HTTP
Environment variables and configuration
Basic security concepts
Git and version control
Terminal/command line usage
DO NOT EXPLAIN:

How to create files or folders
How to install Python packages
How to use pip or virtual environments
Basic Python syntax
How to run Python scripts
CRITICAL REMINDERS:

This is a real production system for datacenter operations
Security and reliability are paramount
Code must be maintainable by other developers
Performance matters (technicians waiting for responses)
Error messages must be actionable
All external API calls can fail, handle gracefully
PROJECT-SPECIFIC CONTEXT TO ALWAYS REMEMBER:

Open WebUI runs on http://216.81.245.140:8080
We use environment variables from .env file
Knowledge Base ID is required for RAG queries
Audio recording uses sounddevice library
Screenshots use macOS screencapture command
Whisper API for speech-to-text
Target users are datacenter technicians using smart glasses
Hands-free operation is critical
Response time should be under 5 seconds when possible
If you need clarification on any requirement, ask a specific question rather than making assumptions.