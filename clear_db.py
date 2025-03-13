#!/usr/bin/env python3
"""
Clear ChromaDB Database

This script clears the ChromaDB collection used by vector_embedder.py
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
    """Load configuration from YAML file."""
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
    """Clear the ChromaDB database."""
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

if __name__ == "__main__":
    logger.info("Starting database clearing process")
    clear_database()
    logger.info("Database clearing process completed") 
