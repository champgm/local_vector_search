#!/usr/bin/env python3
"""
PyTorch CUDA Installation Helper

This script helps install the correct version of PyTorch with CUDA support
based on your system's CUDA version.
"""

import os
import sys
import subprocess
import platform
from typing import Optional

def print_header(text: str):
    """Print a header with the given text."""
    print("\n" + "=" * 70)
    print(f" {text} ".center(70, '='))
    print("=" * 70)

def print_step(step_num: int, text: str):
    """Print a step with the given number and text."""
    print(f"\n[Step {step_num}] {text}")

def run_command(command: str) -> tuple:
    """Run a command and return the output and return code."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=False, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def detect_cuda_version() -> Optional[str]:
    """Detect the installed CUDA version."""
    print_step(1, "Detecting installed CUDA version...")
    
    # Method 1: Check nvcc version
    stdout, stderr, retcode = run_command("nvcc --version")
    if retcode == 0 and "release" in stdout:
        for line in stdout.split('\n'):
            if "release" in line:
                # Parse version like "release 11.8" or "release 12.1"
                parts = line.split("release")[1].strip().split(".")
                if len(parts) >= 2:
                    major, minor = parts[0].strip(), parts[1].strip()
                    return f"{major}.{minor}"
    
    # Method 2: Check nvidia-smi
    stdout, stderr, retcode = run_command("nvidia-smi")
    if retcode == 0 and "CUDA Version:" in stdout:
        for line in stdout.split('\n'):
            if "CUDA Version:" in line:
                version = line.split("CUDA Version:")[1].strip()
                parts = version.split(".")
                if len(parts) >= 2:
                    return f"{parts[0]}.{parts[1]}"
    
    # Method 3: Check for CUDA_PATH or CUDA_HOME environment variables
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        parts = cuda_path.split("\\")[-1].split("v")[-1].split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
    
    return None

def get_pytorch_install_command(cuda_version: str) -> str:
    """Get the PyTorch installation command based on the CUDA version."""
    cuda_major = cuda_version.split('.')[0]
    
    # Map CUDA versions to PyTorch CUDA versions
    cuda_map = {
        "11": "cu118",  # For CUDA 11.x, use cu118
        "12": "cu121",  # For CUDA 12.x, use cu121
    }
    
    cuda_tag = cuda_map.get(cuda_major, "cu121")  # Default to cu121 if unknown
    
    return f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_tag}"

def check_gpu():
    """Check if the system has an NVIDIA GPU."""
    print_step(2, "Checking for NVIDIA GPU...")
    
    if platform.system() == "Windows":
        stdout, stderr, retcode = run_command("wmic path win32_VideoController get name")
        if retcode == 0 and "NVIDIA" in stdout:
            print("✅ NVIDIA GPU detected!")
            return True
    else:
        stdout, stderr, retcode = run_command("lspci | grep -i nvidia")
        if retcode == 0 and stdout:
            print("✅ NVIDIA GPU detected!")
            return True
    
    print("❌ No NVIDIA GPU detected. CUDA acceleration will not be possible.")
    return False

def check_pytorch():
    """Check if PyTorch is installed and if it has CUDA support."""
    print_step(3, "Checking PyTorch installation...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA support is available!")
            print(f"CUDA version used by PyTorch: {torch.version.cuda}")
            return True, True
        else:
            print("❌ PyTorch CUDA support is not available.")
            return True, False
    except ImportError:
        print("❌ PyTorch is not installed.")
        return False, False

def install_pytorch_with_cuda(cuda_version: str):
    """Install PyTorch with CUDA support."""
    print_step(4, f"Installing PyTorch with CUDA {cuda_version} support...")
    
    cmd = get_pytorch_install_command(cuda_version)
    print(f"Running: {cmd}")
    stdout, stderr, retcode = run_command(cmd)
    
    if retcode == 0:
        print("✅ PyTorch with CUDA support has been installed successfully!")
        return True
    else:
        print("❌ Failed to install PyTorch with CUDA support.")
        print("Error:")
        print(stderr)
        return False

def main():
    print_header("PyTorch CUDA Installation Helper")
    
    # Check if system has an NVIDIA GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("\nNo NVIDIA GPU detected. Cannot proceed with CUDA installation.")
        sys.exit(1)
    
    # Check PyTorch installation
    pytorch_installed, has_cuda = check_pytorch()
    
    if pytorch_installed and has_cuda:
        print("\nPyTorch with CUDA support is already installed and working!")
        
        choice = input("\nDo you want to reinstall PyTorch to match your CUDA version? (y/n): ").lower()
        if choice != 'y':
            print("No changes made. You can now run the vector embedding system with GPU support!")
            sys.exit(0)
    
    # Detect CUDA version
    cuda_version = detect_cuda_version()
    if cuda_version:
        print(f"✅ Detected CUDA version: {cuda_version}")
    else:
        print("❌ Could not detect CUDA version.")
        print("Please visit https://developer.nvidia.com/cuda-downloads to install CUDA Toolkit.")
        cuda_version = input("If CUDA is already installed, enter the version manually (e.g., 11.8, 12.1): ")
        if not cuda_version:
            print("Cannot proceed without CUDA version. Exiting.")
            sys.exit(1)
    
    # Install PyTorch with CUDA support
    success = install_pytorch_with_cuda(cuda_version)
    
    if success:
        print_header("Installation Complete")
        print("Now run check_gpu.py to verify that GPU acceleration is working!")
    else:
        print_header("Installation Failed")
        print("Please try these steps:")
        print("1. Make sure NVIDIA drivers are installed")
        print("2. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
        print("3. Try manually running:")
        print(f"   {get_pytorch_install_command(cuda_version)}")

if __name__ == "__main__":
    main() 
