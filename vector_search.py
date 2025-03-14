#!/usr/bin/env python3
"""
Vector Search Script

This script prompts for user input, generates an embedding, and performs a
vector similarity search in the ChromaDB database created by vector_embedder.py.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sys
import yaml
import logging
import chromadb
import openai
from pathlib import Path
from typing import Dict, List, Any, Optional
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('vector_search')

class VectorSearch:
    """Handles searching for similar documents in the vector database."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration from YAML file."""
        self.config = self._load_config(config_path)
        self.openai_client = self._init_openai_client()
        self.chroma_client, self.collection = self._init_chromadb()
        self.tiktoken_encoder = self._init_tiktoken_encoder()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
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
            
    def _init_openai_client(self) -> Any:
        """Initialize OpenAI client with API key from config."""
        try:
            api_key = self.config['openai']['api_key']
            if api_key == "your-openai-api-key-here":
                logger.error("Please update the OpenAI API key in config.yaml")
                sys.exit(1)
            # For OpenAI version 0.28.1
            openai.api_key = api_key
            return openai
        except KeyError:
            logger.error("OpenAI API key not found in configuration")
            sys.exit(1)
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            db_path = self.config['chromadb']['path']
            collection_name = self.config['chromadb']['collection_name']
            
            # Initialize persistent client
            client = chromadb.PersistentClient(path=db_path)
            
            # Get collection
            try:
                collection = client.get_collection(name=collection_name)
                logger.info(f"Connected to collection: {collection_name}")
                return client, collection
            except:
                logger.error(f"Collection '{collection_name}' not found. Run vector_embedder.py first to create it.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            sys.exit(1)
    
    def _init_tiktoken_encoder(self):
        """Initialize tiktoken encoder for token counting."""
        try:
            model = self.config['openai']['model']
            # Get the encoding for the specified model
            return tiktoken.encoding_for_model(model)
        except Exception as e:
            logger.warning(f"Could not get specific encoding for model {model}: {e}")
            # Fall back to cl100k_base encoding which is used by most newer models
            return tiktoken.get_encoding("cl100k_base")
    
    def truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to specified maximum token count."""
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
            
            logger.info(f"Query truncated from {token_count} to {max_tokens} tokens")
            return truncated_text
        except Exception as e:
            logger.error(f"Error truncating text: {e}")
            # Return original text if truncation fails
            return text
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using OpenAI API."""
        try:
            model = self.config['openai']['model']
            max_tokens = self.config['openai'].get('maximum_tokens', 8192)
            
            # Truncate text to maximum token count
            truncated_text = self.truncate_text_by_tokens(text, max_tokens)
            
            # For OpenAI version 0.28.1
            response = self.openai_client.Embedding.create(
                input=truncated_text, 
                model=model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def search(self, query_text: str, n_results: int = 5) -> Optional[Dict]:
        """
        Search for similar documents based on the query text.
        
        Args:
            query_text: The text to search for
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results or None if search fails
        """
        try:
            # Generate embedding for the query
            query_embedding = self.get_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return None
                
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return results
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return None
    
    def display_results(self, results: Dict, show_preview: bool = False) -> None:
        """
        Display search results in a readable format.
        
        Args:
            results: Dictionary containing search results from ChromaDB
            show_preview: Whether to show a preview of the document content
        """
        if not results or not results['ids'] or not results['ids'][0]:
            print("No results found.")
            return
            
        print(f"\nFound {len(results['ids'][0])} results:")
        print("-" * 80)
        
        for i, (doc_id, metadata, distance) in enumerate(
            zip(results['ids'][0], results['metadatas'][0], results['distances'][0])
        ):
            similarity = 1 - distance  # Convert distance to similarity score
            
            print(f"{i+1}. {doc_id}")
            print(f"   Path: {metadata.get('absolute_path', 'N/A')}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   File type: {metadata.get('file_type', 'N/A')}")
            print(f"   File size: {metadata.get('file_size', 0) / 1024:.2f} KB")
            
            if show_preview and results.get('documents') and results['documents'][0]:
                preview = results['documents'][0][i]
                preview = preview.replace('\n', ' ')[:100] + '...'
                print(f"   Preview: {preview}")
                
            print("-" * 80)

def main():
    """Main entry point."""
    searcher = VectorSearch()
    
    print("\nVector Search")
    print("=" * 50)
    print("Enter your search query (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\nQuery> ").strip()
            if not query:
                continue
                
            if query.lower() in ('quit', 'exit', 'q'):
                break
                
            # Perform search
            n_results = 5  # Default number of results
            results = searcher.search(query, n_results)
            
            if results:
                searcher.display_results(results, show_preview=True)
            else:
                print("Search failed or returned no results.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    print("\nThank you for using Vector Search!")
    
if __name__ == "__main__":
    main() 
