"""
Colab setup script for langchain business plan generator
"""
print("Setting up Colab environment...")

# Install dependencies quietly
!pip install -q transformers==4.30.0 torch==2.0.1 peft==0.4.0 datasets==2.12.0
!pip install -q streamlit==1.24.0 huggingface-hub==0.16.0 accelerate==0.20.0
!pip install -q bitsandbytes==0.41.0 sentencepiece==0.1.99
!pip install -q python-docx==0.8.11

import subprocess
import sys
import torch

def check_gpu():
    """Check if GPU is available and print device info"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nGPU is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of devices: {torch.cuda.device_count()}")
        # Test GPU with a small tensor operation
        x = torch.randn(3, 3).to(device)
        print("\nGPU test successful!")
    else:
        print("\nNo GPU found. Please make sure you're using a GPU runtime in Colab")
        print("Runtime → Change runtime type → Hardware accelerator → GPU")
        sys.exit(1)

if __name__ == "__main__":
    print("\nChecking GPU availability...")
    check_gpu()

    # Print torch version and cuda availability for debugging
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

    print("\nSetup complete! You can now proceed with model training.")