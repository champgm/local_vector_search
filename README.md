# Local Vector Search

A set of tools for creating and managing a local vector database of text documents using OpenAI embeddings and ChromaDB.

## Overview

This project enables semantic search over your text documents by:
1. Scanning directories for text files
2. Generating vector embeddings using OpenAI's API
3. Storing these embeddings in a local ChromaDB database
4. Enabling vector similarity search

## Components

The system consists of three main scripts:

### `vector_embedder.py`

The main script for traversing directories, generating embeddings for text files, and storing them in ChromaDB.

**Features:**
- Recursively traverses directories to process text files
- Generates vector embeddings using OpenAI's embedding models
- Stores embeddings in ChromaDB for efficient similarity search
- Skips previously processed files that haven't changed
- Configurable via YAML configuration file

**Usage:**
```
python vector_embedder.py
```

### `clear_documents.py`

A utility script to remove all documents from the ChromaDB collection without deleting the collection itself.

**Features:**
- Preserves the ChromaDB collection structure
- Removes all document embeddings and metadata
- Useful for refreshing your database while keeping the same structure

**Usage:**
```
python clear_documents.py
```

### `clear_db.py`

A utility script to completely reset the ChromaDB collection by deleting and recreating it.

**Features:**
- More aggressive reset than `clear_documents.py`
- Deletes the entire collection and recreates it
- Useful for fixing corrupted databases or starting fresh

**Usage:**
```
python clear_db.py
```

## Setup

### Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

#### Windows

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Deactivate when done
deactivate
```

#### macOS/Linux

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

### Configuration

1. Copy `config.template.yaml` to `config.yaml` and update it with your settings:
   ```
   cp config.template.yaml config.yaml
   ```

2. Update the configuration file with:
   - Your OpenAI API key
   - ChromaDB storage location
   - File processing preferences (extensions, excluded directories, etc.)

3. Install the required dependencies:
   ```
   # Make sure your virtual environment is activated
   pip install -r requirements.txt
   ```

## Configuration

The `config.yaml` file contains several important settings:

```yaml
openai:
  api_key: "your-openai-api-key-here"
  model: "text-embedding-ada-002"
  maximum_tokens: 8192

chromadb:
  path: "./chroma_db"
  collection_name: "document_embeddings"

file_processing:
  text_extensions: [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"]
  exclude_dirs: [".git", "__pycache__", "node_modules", "venv", ".venv", "env"]
  max_file_size_kb: 1024
```

## License

[Specify your license here]

## Contributing

[Contribution guidelines]
