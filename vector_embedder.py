#!/usr/bin/env python3
"""
Vector Embedder Script

This script traverses directories, generates vector embeddings for text-based files,
and stores them in a ChromaDB database for vector similarity search.

Features:
- Recursively traverses directories to process text files
- Generates vector embeddings using Hugging Face's embedding models
- Stores embeddings in ChromaDB for efficient similarity search
- Skips previously processed files that haven't changed
- Configurable via YAML configuration file

Usage:
    python vector_embedder.py

Requirements:
    - Hugging Face Transformers library
    - ChromaDB for vector storage
    - See requirements.txt for all dependencies
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys
import yaml
import logging
import hashlib
import chromadb
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import tiktoken
from pathlib import Path
from typing import Dict, List, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('vector_embedder')

class VectorEmbedder:
    """
    Handles embedding generation and storage for files.
    
    This class is responsible for traversing directories, processing text files,
    generating vector embeddings via Hugging Face, and storing them in ChromaDB.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        embedding_model: Hugging Face embedding model instance
        chroma_client: ChromaDB client instance
        collection: ChromaDB collection for storing embeddings
        project_dir (Path): Current working directory
        parent_dir (Path): Parent directory to start traversal from
        tiktoken_encoder: Tokenizer for counting and truncating tokens
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the VectorEmbedder with configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
                               Defaults to "config.yaml" in the current directory
        
        Raises:
            SystemExit: If the configuration file is missing or invalid
        """
        self.config = self._load_config(config_path)
        self.embedding_model = self._init_embedding_model()
        self.chroma_client, self.collection = self._init_chromadb()
        self.project_dir = Path.cwd()
        self.parent_dir = self.project_dir.parent
        self.tiktoken_encoder = self._init_tiktoken_encoder()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            SystemExit: If the file is not found or contains invalid YAML
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            logger.info(f"Please copy config.template.yaml to {config_path} and update it with your settings")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            sys.exit(1)
            
    def _init_embedding_model(self) -> Any:
        """
        Initialize Hugging Face embedding model.
        
        Returns:
            Any: Hugging Face SentenceTransformer model instance
            
        Raises:
            SystemExit: If model initialization fails
        """
        try:
            model_name = self.config['huggingface']['model']
            logger.info(f"Loading embedding model: {model_name}")
            
            # Check if CUDA (GPU) is available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Load the model
            model = SentenceTransformer(model_name)
            
            # Move model to GPU if available
            model = model.to(device)
            
            return model
        except KeyError:
            logger.error("Hugging Face model name not found in configuration")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error initializing Hugging Face model: {e}")
            sys.exit(1)
    
    def _init_chromadb(self):
        """
        Initialize ChromaDB client and collection.
        
        Returns:
            tuple: (ChromaDB client, ChromaDB collection)
            
        Raises:
            SystemExit: If ChromaDB initialization fails
        """
        try:
            db_path = self.config['chromadb']['path']
            collection_name = self.config['chromadb']['collection_name']
            
            # Ensure the directory exists
            os.makedirs(db_path, exist_ok=True)
            
            # Initialize persistent client
            client = chromadb.PersistentClient(path=db_path)
            
            # Get or create collection
            try:
                collection = client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except:
                collection = client.create_collection(name=collection_name)
                logger.info(f"Created new collection: {collection_name}")

            # Before we start, let's vacuum the database
            collection.vacuum()
                
            return client, collection
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            sys.exit(1)
    
    def _get_file_id(self, file_path: Path) -> str:
        """
        Generate a unique ID for a file based on its relative path.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            str: Unique identifier for the file
        """
        rel_path = file_path.relative_to(self.parent_dir)
        return str(rel_path)
    
    def _is_text_file(self, file_path: Path) -> bool:
        """
        Check if a file is a text-based file based on its extension.
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            bool: True if the file is a text file, False otherwise
        """
        extensions = self.config['file_processing']['text_extensions']
        return file_path.suffix.lower() in extensions
    
    def _should_exclude_dir(self, dir_path: Path) -> bool:
        """
        Check if a directory should be excluded from traversal.
        
        This helps skip directories like .git, __pycache__, node_modules, etc.
        
        Args:
            dir_path (Path): Path to the directory
            
        Returns:
            bool: True if the directory should be excluded, False otherwise
        """
        exclude_dirs = self.config['file_processing']['exclude_dirs']
        
        # Skip the project directory itself
        if dir_path.samefile(self.project_dir):
            return True
            
        # Skip directories in the exclude list
        for exclude in exclude_dirs:
            if exclude in dir_path.parts:
                return True
                
        return False
    
    def _init_tiktoken_encoder(self):
        """
        Initialize tiktoken encoder for token counting.
        
        Returns:
            tiktoken.Encoding: Encoder instance for the specified model
        """
        try:
            model = self.config['huggingface']['model']
            # Get the encoding for the specified model
            return tiktoken.encoding_for_model(model)
        except Exception as e:
            logger.warning(f"Could not get specific encoding for model {model}: {e}")
            # Fall back to cl100k_base encoding which is used by most newer models
            return tiktoken.get_encoding("cl100k_base")
    
    def truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to specified maximum token count.
        
        This ensures text stays within OpenAI's token limits.
        
        Args:
            text (str): Text to truncate
            max_tokens (int): Maximum number of tokens allowed
            
        Returns:
            str: Truncated text if it exceeds token limit, original text otherwise
        """
        try:
            # Encode the text to tokens
            tokens = self.tiktoken_encoder.encode(text)
            token_count = len(tokens)
            
            # If the text is already within token limit, return it unchanged
            if token_count <= max_tokens:
                return text
                
            # Truncate tokens and decode back to string
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.tiktoken_encoder.decode(truncated_tokens)
            
            logger.info(f"Text truncated from {token_count} to {max_tokens} tokens")
            return truncated_text
        except Exception as e:
            logger.error(f"Error truncating text: {e}")
            # Return original text if truncation fails
            return text
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Hugging Face model.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            max_tokens = self.config['huggingface'].get('maximum_tokens', 8192)
            
            # Truncate text to maximum token count
            truncated_text = self.truncate_text_by_tokens(text, max_tokens)
            
            # Get the device from the model
            device = next(self.embedding_model.parameters()).device
            
            # Generate embedding using Hugging Face model
            with torch.no_grad():
                embedding = self.embedding_model.encode(truncated_text)
                
            # Convert to list and return
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def process_file(self, file_path: Path) -> bool:
        """
        Process a single file and store its embedding in ChromaDB.
        
        This function:
        1. Checks if the file is too large
        2. Checks if the file has changed since last processing
        3. Reads the file content
        4. Generates embedding
        5. Stores embedding and metadata in ChromaDB
        
        Args:
            file_path (Path): Path to the file to process
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            # Skip if file is too large
            max_size_kb = self.config['file_processing']['max_file_size_kb']
            if file_path.stat().st_size > max_size_kb * 1024:
                logger.warning(f"Skipping large file: {file_path} ({file_path.stat().st_size / 1024:.2f} KB)")
                return False
                
            # Generate a unique ID for the file
            file_id = self._get_file_id(file_path)
            
            # Check if file is already in the database
            existing_ids = self.collection.get(ids=[file_id])
            if existing_ids['ids'] and not self._has_file_changed(file_path, existing_ids):
                logger.debug(f"Skipping unchanged file: {file_path}")
                return False
                
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Log token count
            token_count = len(self.tiktoken_encoder.encode(content))
            max_tokens = self.config['huggingface'].get('maximum_tokens', 8192)
            if token_count > max_tokens:
                logger.info(f"File will be truncated: {file_path} ({token_count} tokens > {max_tokens} max)")
            else:
                logger.debug(f"File token count: {file_path} ({token_count} tokens)")
            
            # Generate embedding
            embedding = self.get_embedding(content)
            if not embedding:
                return False
                
            # Store in ChromaDB
            metadata = {
                "absolute_path": str(file_path.absolute()),
                "relative_path": str(file_path.relative_to(self.parent_dir)),
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "modified_time": file_path.stat().st_mtime,
                "token_count": token_count
            }
            
            # Add or update in collection
            self.collection.upsert(
                ids=[file_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[content[:1000]]  # Store first 1000 chars as preview
            )
            
            logger.debug(f"Processed file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def _has_file_changed(self, file_path: Path, existing_record: Dict) -> bool:
        """
        Check if a file has changed since it was last processed.
        
        Compares the file's modification time with the stored metadata.
        
        Args:
            file_path (Path): Path to the file
            existing_record (Dict): Record retrieved from ChromaDB
            
        Returns:
            bool: True if the file has changed, False otherwise
        """
        if not existing_record['metadatas']:
            return True
            
        metadata = existing_record['metadatas'][0]
        return metadata.get('modified_time', 0) != file_path.stat().st_mtime
    
    def traverse_directories(self) -> Dict[str, int]:
        """
        Traverse directories and process files.
        
        This function walks through the parent directory and its subdirectories,
        identifying text files, and processing each one.
        
        Returns:
            Dict[str, int]: Statistics about the processing operation
        """
        stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total": 0
        }
        
        # Get all text files in parent directory and its subdirectories
        logger.info(f"Starting traversal from parent directory: {self.parent_dir}")
        all_files = []
        
        for root, dirs, files in os.walk(self.parent_dir):
            root_path = Path(root)
            
            # Skip excluded directories
            if self._should_exclude_dir(root_path):
                dirs.clear()  # Don't traverse into this directory
                continue
                
            # Process files in this directory
            for file in files:
                file_path = root_path / file
                if self._is_text_file(file_path):
                    all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} text files to process")
        
        # Process files with progress bar
        for file_path in tqdm(all_files, desc="Processing files"):
            stats["total"] += 1
            result = self.process_file(file_path)
            
            if result:
                stats["processed"] += 1
            else:
                stats["skipped"] += 1
                
        return stats

def main():
    """
    Main entry point for the vector embedding process.
    
    Initializes the VectorEmbedder, traverses directories, and reports statistics.
    """

    logger.info("Starting vector embedding process")
    embedder = VectorEmbedder()
    stats = embedder.traverse_directories()
    
    logger.info(f"Vector embedding process completed:")
    logger.info(f"  Total files found: {stats['total']}")
    logger.info(f"  Files processed: {stats['processed']}")
    logger.info(f"  Files skipped: {stats['skipped']}")
    logger.info(f"  Files failed: {stats['failed']}")
    
if __name__ == "__main__":
    main() 
