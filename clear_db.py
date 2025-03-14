#!/usr/bin/env python3
"""
Clear ChromaDB Database

This script clears the ChromaDB collection used by vector_embedder.py by completely
deleting the collection and recreating it from scratch. This effectively removes
all vector embeddings, document references, and metadata from the database.

Use this script when you want to:
- Reset your vector database completely
- Fix issues with corrupted data
- Start a fresh embedding process

Usage:
    python clear_db.py

Requirements:
    - ChromaDB (for vector storage)
    - Configuration file (config.yaml) with ChromaDB settings
"""

import yaml
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import logging
import chromadb
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('clear_db')

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

def clear_database():
    """
    Clear the ChromaDB database by deleting and recreating the collection.
    
    This function:
    1. Loads the configuration from config.yaml
    2. Connects to the ChromaDB instance
    3. Deletes the specified collection completely
    4. Creates a new empty collection with the same name
    
    This is a more aggressive option than clear_documents.py, which only 
    removes documents but keeps the collection structure intact.
    
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
        
        # Check if collection exists
        try:
            # Delete the collection if it exists
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            
            # Recreate an empty collection
            client.create_collection(name=collection_name)
            logger.info(f"Created new empty collection: {collection_name}")

            
        except Exception as e:
            logger.error(f"Collection does not exist or could not be deleted: {e}")
    
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
    
    # now, vacuum the database
    client.vacuum()
    logger.info("Database vacuumed")

if __name__ == "__main__":
    logger.info("Starting database clearing process")
    clear_database()
    logger.info("Database clearing process completed") 
