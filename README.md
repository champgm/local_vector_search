# Local Vector Search

A Python tool for embedding and searching local files using vector embeddings.

## Features

- Traverses directories to identify text-based files
- Generates vector embeddings for each file using OpenAI's embedding models
- Stores embeddings in a ChromaDB database for efficient vector similarity search
- Tracks file changes to avoid re-embedding unchanged files
- Configurable via YAML configuration file

## Setup

1. Create and activate the virtual environment:

```bash
# If you haven't already created the virtual environment:
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your configuration:

```bash
# Copy the template configuration file
cp config.template.yaml config.yaml

# Edit the configuration file with your own settings
# Be sure to add your OpenAI API key
```

## Usage

Run the script to start embedding files:

```bash
python vector_embedder.py
```

The script will:
1. Traverse the parent directory (excluding this project directory)
2. Find all text-based files (according to the extensions in your config)
3. Generate and store embeddings for each file
4. Skip files that haven't changed since the last run

## Configuration

The `config.yaml` file contains several configuration options:

- **OpenAI API settings**: API key and embedding model to use
- **ChromaDB settings**: Where to store the database and the collection name
- **File processing settings**: 
  - Which file extensions to consider as text files
  - Maximum file size to process
  - Directories to exclude from traversal

## Notes

- The ChromaDB database is stored locally in the path specified in your config
- Large files are skipped by default (configurable threshold)
- The script is designed to be run multiple times, only processing new or changed files 
