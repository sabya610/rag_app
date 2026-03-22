# rag_meeting_app

## Overview
This application reads meeting transcripts, extracts key meeting pointers (issues, decisions, action items, blockers, etc.), and indexes them for semantic search using local LLM and embedding models.

## Features
- Reads transcripts from `data/raw/transcripts/`
- Extracts structured meeting issues using a local LLM
- Embeds and indexes issues for semantic search

## Setup

### 1. Clone the repository
```
git clone <your-repo-url>
cd rag_meeting_app
```

### 2. Install Python dependencies
It is recommended to use a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Models
- Place the embedding model (e.g., all-MiniLM-L6-v2) in `./models/embedding/`
- Place the Llama model (e.g., llama-2-7b-chat.Q4_K_M.gguf) in `./models/`

### 4. Run the Application
```
python -m app.main
```

## Configuration
- By default, the app reads `data/raw/transcripts/meetings_1.txt`. Update the path in `app/main.py` to process other transcripts.

## Usage: Process a Specific Transcript

To process a specific transcript file, provide the path as a command-line argument:

```
python -m app.main path/to/your_transcript.txt
```

If no path is provided, the app defaults to `data/raw/transcripts/meetings_1.txt`.

## Output
- Indexed issues are printed to the console. Extend the app to save or display results as needed.

## Requirements
- Python 3.9+
- Sufficient RAM for Llama model

## Troubleshooting
- Ensure all models are downloaded and placed in the correct folders.
- If you encounter missing package errors, check `requirements.txt`.

## License
MIT
