#!/usr/bin/env python3
"""
Clear Documents in ChromaDB Collection

This script removes all documents from the ChromaDB collection without deleting the collection itself.
This is useful when you want to refresh the document embeddings without losing the collection structure.

Usage:
    python clear_documents.py

Requirements:
    - ChromaDB (for vector storage)
    - Configuration file (config.yaml) with ChromaDB settings
"""

import yaml
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('clear_documents')

def load_config(config_path: str = "config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
                          Defaults to "config.yaml" in the current directory
        
    Returns:
        dict: Configuration dictionary
        
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
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        sys.exit(1)

def clear_documents():
    """
    Clear all documents in the ChromaDB collection.
    
    This function:
    1. Loads the configuration from config.yaml
    2. Connects to the ChromaDB instance
    3. Retrieves the specified collection
    4. Gets all document IDs in the collection
    5. Deletes all documents while preserving the collection structure
    
    Returns:
        None
    """
    config = load_config()
    
    try:
        db_path = config['chromadb']['path']
        collection_name = config['chromadb']['collection_name']
        
        # Ensure the directory exists
        if not os.path.exists(db_path):
            logger.info(f"Database directory does not exist: {db_path}")
            return
        
        # Initialize persistent client
        client = chromadb.PersistentClient(path=db_path)
        
        # Get the collection if it exists
        try:
            collection = client.get_collection(name=collection_name)
            
            # Get all document IDs
            result = collection.get()
            document_ids = result.get('ids', [])
            
            if document_ids:
                # Delete all documents
                collection.delete(ids=document_ids)
                logger.info(f"Deleted {len(document_ids)} documents from collection {collection_name}")
            else:
                logger.info(f"Collection {collection_name} is already empty")
                
        except Exception as e:
            logger.error(f"Could not access collection: {e}")
    
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")

if __name__ == "__main__":
    logger.info("Starting document clearing process")
    clear_documents()
    logger.info("Document clearing process completed") 
