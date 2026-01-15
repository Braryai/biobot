# BioBot Documentation

This directory contains datacenter documentation used by the knowledge base.

## Contents

Your datacenter documentation files should be placed here:
- Switch configuration guides
- Server setup procedures  
- Cabling standards and color codes
- Rack layout diagrams
- Network troubleshooting guides
- IP addressing schemas
- VLAN configurations
- And more...

## Organization

Recommended structure:
```
docs/
├── networking/
│   ├── switches/
│   ├── vlans/
│   └── cabling/
├── servers/
│   ├── dell/
│   ├── hp/
│   └── supermicro/
├── storage/
├── troubleshooting/
└── procedures/
```

## Uploading to Knowledge Base

1. Log into Open WebUI at your server URL
2. Go to **Workspace** → **Knowledge**
3. Select your datacenter knowledge base (or create one)
4. Upload documentation files
5. Wait for indexing to complete
6. Copy the Knowledge Base ID to `biobot-client/config.py`

## Supported Formats

- PDF (.pdf)
- Markdown (.md)
- Text (.txt)
- Word (.docx)
- PowerPoint (.pptx)
- Excel (.xlsx)

## Best Practices

- Use clear, descriptive filenames
- Keep documents focused on specific topics
- Include version numbers or dates in filenames
- Update knowledge base when documents change
- Remove outdated documents
