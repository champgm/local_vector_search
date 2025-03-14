#!/usr/bin/env python3
"""
GPU Compatibility Check Script

This script checks whether your system is properly configured to use GPU acceleration
for the vector embedding system by testing PyTorch, CUDA, and the SentenceTransformer model.
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('gpu_checker')

def load_config(config_path: str = "config.yaml"):
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

def check_gpu_compatibility():
    """Run a series of checks to verify GPU compatibility."""
    
    logger.info("Starting GPU compatibility check")
    
    # Step 1: Check if PyTorch is installed
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Step 2: Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        logger.warning("CUDA is not available. The system will use CPU for processing.")
        logger.info("To enable GPU support, you need to:")
        logger.info("1. Install NVIDIA CUDA Toolkit (compatible with your PyTorch version)")
        logger.info("2. Install the correct NVIDIA drivers for your GPU")
        logger.info("3. Ensure PyTorch is installed with CUDA support")
        return False
    
    # Step 3: Check CUDA version
    cuda_version = torch.version.cuda
    logger.info(f"CUDA version: {cuda_version}")
    
    # Step 4: Check available GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
        logger.info(f"GPU {i}: {gpu_name} (Memory: {gpu_memory:.2f} GB)")
    
    # Step 5: Try to load the model from config
    try:
        config = load_config()
        model_name = config['huggingface']['model']
        logger.info(f"Testing model loading: {model_name}")
        
        # Load the model first on CPU to prevent any immediate CUDA errors
        model_cpu = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully on CPU")
        
        # Now try to move it to GPU
        device = torch.device('cuda')
        logger.info(f"Moving model to {device}")
        
        start_time = time.time()
        model_gpu = model_cpu.to(device)
        load_time = time.time() - start_time
        logger.info(f"Model successfully moved to GPU in {load_time:.2f} seconds")
        
        # Step 6: Test a simple embedding generation
        test_text = "This is a test sentence to verify GPU acceleration is working."
        logger.info(f"Testing embedding generation on GPU with text: '{test_text}'")
        
        # Run on CPU for comparison
        start_time = time.time()
        with torch.no_grad():
            _ = model_cpu.encode(test_text)
        cpu_time = time.time() - start_time
        logger.info(f"CPU embedding generation took {cpu_time:.4f} seconds")
        
        # Run on GPU
        start_time = time.time()
        with torch.no_grad():
            _ = model_gpu.encode(test_text)
        gpu_time = time.time() - start_time
        logger.info(f"GPU embedding generation took {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        logger.info(f"GPU speedup factor: {speedup:.2f}x")
        
        if speedup < 1:
            logger.warning("GPU processing is slower than CPU. This could be due to:")
            logger.warning("1. The model or input is too small to benefit from GPU acceleration")
            logger.warning("2. GPU overhead for small operations")
            logger.warning("3. GPU initialization time included in the measurement")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        logger.error("GPU compatibility check failed")
        return False

def print_summary(compatible):
    """Print a summary of the compatibility check."""
    
    logger.info("\n" + "="*50)
    logger.info("GPU COMPATIBILITY SUMMARY")
    logger.info("="*50)
    
    if compatible:
        logger.info("✅ Your system is correctly configured for GPU acceleration!")
        logger.info("   The vector embedding system should use your GPU automatically.")
    else:
        logger.info("❌ Your system is not fully configured for GPU acceleration.")
        logger.info("   The vector embedding system will fall back to CPU processing.")
        logger.info("\nTo enable GPU support, make sure you have:")
        logger.info("1. A compatible NVIDIA GPU")
        logger.info("2. NVIDIA CUDA Toolkit installed (compatible with your PyTorch version)")
        logger.info("3. Appropriate NVIDIA drivers installed")
        logger.info("4. PyTorch installed with CUDA support (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121)")
    
    logger.info("="*50)

def main():
    """Main entry point."""
    compatible = check_gpu_compatibility()
    print_summary(compatible)

if __name__ == "__main__":
    main() 
