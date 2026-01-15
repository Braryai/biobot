#!/bin/bash
# BioBot Setup Script
# Quick setup for BioBot Voice Client

set -e  # Exit on error

echo "============================================================"
echo "🤖 BioBot Voice Client - Setup Script"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "biobot_voice.py" ]; then
    echo -e "${RED}❌ Error: biobot_voice.py not found${NC}"
    echo "Please run this script from the biobot-client directory:"
    echo "  cd biobot-client"
    echo "  ./setup.sh"
    exit 1
fi

echo "📂 Current directory: $(pwd)"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${YELLOW}⚠ Python $PYTHON_VERSION found, but 3.11+ recommended${NC}"
    fi
else
    echo -e "${RED}❌ Python 3 not found${NC}"
    echo "Please install Python 3.11 or higher:"
    echo "  brew install python@3.11"
    exit 1
fi
echo ""

# Check if config.py exists
echo "⚙️  Checking configuration..."
if [ ! -f "config.py" ]; then
    echo -e "${YELLOW}⚠ config.py not found${NC}"
    echo "Creating from template..."
    cp config.py.example config.py
    echo -e "${GREEN}✓ Created config.py${NC}"
    echo -e "${YELLOW}⚠ You need to edit config.py with your credentials!${NC}"
    echo ""
    echo "Required settings:"
    echo "  - OPENWEBUI_TOKEN (from Open WebUI → Settings → API Keys)"
    echo "  - KNOWLEDGE_ID (from Open WebUI → Workspace → Knowledge)"
    echo "  - OPENAI_API_KEY (from https://platform.openai.com/api-keys)"
    echo ""
    read -p "Press Enter to edit config.py now, or Ctrl+C to exit and edit later..."
    ${EDITOR:-nano} config.py
else
    echo -e "${GREEN}✓ config.py found${NC}"
fi
echo ""

# Install dependencies
echo "📦 Installing Python dependencies..."
if pip3 install -r requirements.txt; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    echo "Try:"
    echo "  pip3 install --upgrade pip"
    echo "  pip3 install -r requirements.txt"
    exit 1
fi
echo ""

# Check permissions
echo "🔐 Checking macOS permissions..."
echo ""
echo "BioBot requires the following permissions:"
echo "  1. Microphone access (for audio recording)"
echo "  2. Screen Recording (for screenshot capture)"
echo ""
echo "You may see permission prompts when you first run BioBot."
echo "Grant these permissions in System Settings → Privacy & Security"
echo ""

# Test Open WebUI connection
echo "🔗 Testing Open WebUI connection..."
if [ -f "config.py" ]; then
    # Extract URL from config.py
    OPENWEBUI_URL=$(grep "OPENWEBUI_URL" config.py | cut -d'"' -f2)
    
    if [ -n "$OPENWEBUI_URL" ]; then
        if curl -s --max-time 5 "$OPENWEBUI_URL/api/config" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Open WebUI is accessible at $OPENWEBUI_URL${NC}"
        else
            echo -e "${YELLOW}⚠ Could not connect to Open WebUI at $OPENWEBUI_URL${NC}"
            echo "  Please verify:"
            echo "  - Open WebUI is running"
            echo "  - URL in config.py is correct"
            echo "  - You have network access to the server"
        fi
    else
        echo -e "${YELLOW}⚠ OPENWEBUI_URL not configured yet${NC}"
    fi
fi
echo ""

# All done
echo "============================================================"
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Verify your configuration:"
echo "   nano config.py"
echo ""
echo "2. Make sure you have set:"
echo "   - OPENWEBUI_TOKEN"
echo "   - KNOWLEDGE_ID"
echo "   - OPENAI_API_KEY"
echo ""
echo "3. Run BioBot:"
echo "   python3 biobot_voice.py"
echo ""
echo "4. Test with:"
echo "   - Right Shift: Audio-only mode"
echo "   - Right Command: Audio + Screenshot mode"
echo ""
echo "See README.md for full documentation."
echo ""
echo "Happy datacenter troubleshooting! 🤖"
echo ""
