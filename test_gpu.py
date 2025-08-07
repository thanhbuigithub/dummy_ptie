#!/usr/bin/env python3
"""
Test script to verify GPU functionality and PyTorch installation
"""

import torch
import numpy as np

def test_gpu():
    print("=== GPU Test Results ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        
        # Test GPU computation
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.mm(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"GPU matrix multiplication test: {elapsed_time:.2f} ms")
        print("GPU test PASSED!")
    else:
        print("CUDA not available - using CPU only")
        print("GPU test FAILED!")

if __name__ == "__main__":
    test_gpu() 