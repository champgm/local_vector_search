# Local Vector Search

A set of tools for creating and managing a local vector database of text documents using Hugging Face's SFR-Embedding-Mistral model and ChromaDB.

## Overview

This project enables semantic search over your text documents by:
1. Scanning directories for text files
2. Generating vector embeddings using Salesforce's SFR-Embedding-Mistral model
3. Storing these embeddings in a local ChromaDB database
4. Enabling vector similarity search

## Components

The system consists of three main scripts:

### `vector_embedder.py`

The main script for traversing directories, generating embeddings for text files, and storing them in ChromaDB.

**Features:**
- Recursively traverses directories to process text files
- Generates vector embeddings using Salesforce's SFR-Embedding-Mistral model
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

### `vector_search.py`

A script for performing vector similarity searches on your embedded documents.

**Features:**
- Interactive prompt for entering search queries
- Uses the same Hugging Face embedding model for query embedding
- Returns the most semantically similar documents
- Displays relevance scores and document metadata

**Usage:**
```
python vector_search.py
```

### `test_dependencies.py`

A utility script to verify that all dependencies are working correctly.

**Features:**
- Tests importing all required packages
- Verifies basic functionality of each dependency
- Useful for debugging after updating dependencies

**Usage:**
```
python test_dependencies.py
```

### `vacuum_db.py`

A utility script to vacuum the SQLite database used by ChromaDB.

**Features:**
- Optimizes the database after upgrading ChromaDB versions
- Rebuilds the database file to reclaim space
- Useful when upgrading from ChromaDB versions below 0.5.6

**Usage:**
```
python vacuum_db.py
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
   - ChromaDB storage location
   - File processing preferences (extensions, excluded directories, etc.)

3. Install the required dependencies:
   ```
   # Make sure your virtual environment is activated
   pip install -r requirements.txt
   ```

## Dependencies

This project uses the following dependencies:
- ChromaDB 0.6.3 - Vector database for storing embeddings
- Sentence Transformers 2.3.1 - For generating embeddings with Hugging Face models
- Transformers 4.39.3 - Hugging Face's transformer models
- PyTorch 2.2.1 - Deep learning framework
- tiktoken 0.9.0 - Tokenizer for text models
- PyYAML 6.0.1 - For configuration file parsing
- tqdm 4.66.1 - For progress bars
- python-dotenv 1.0.1 - For environment variable management

To verify all dependencies are working correctly, run:
```
python test_dependencies.py
```

## Configuration

The `config.yaml` file contains several important settings:

```yaml
huggingface:
  model: "Salesforce/SFR-Embedding-Mistral"
  maximum_tokens: 8192

chromadb:
  path: "./chroma_db"
  collection_name: "file_embeddings"

file_processing:
  text_extensions: [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"]
  exclude_dirs: [".git", "__pycache__", "node_modules", "venv", ".venv", "env"]
  max_file_size_kb: 1024000
```

## License

[Specify your license here]

## Contributing

[Contribution guidelines]
