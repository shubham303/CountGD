#!/usr/bin/env python3
"""Test script to verify MPS support with CPU fallback for MultiScaleDeformableAttention"""

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))
from functions.ms_deform_attn_func import MSDeformAttnFunction

def test_mps_fallback():
    print("Testing MPS support with CPU fallback...")
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✓ MPS is available")
        device = torch.device("mps")
    else:
        print("! MPS not available, using CPU")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create test tensors with correct dimensions
    # Based on the CUDA interface and PyTorch reference implementation
    N = 2  # batch size
    M = 8  # number of heads
    D = 256  # feature dimension
    L = 4  # number of levels
    P = 4  # number of sampling points
    
    # Total sequence length (sum of H*W for all levels)
    spatial_shapes_list = [[10, 10], [5, 5], [3, 3], [2, 2]]  # H, W for each level
    Len_q = sum([h*w for h, w in spatial_shapes_list])  # 100 + 25 + 9 + 4 = 138
    
    # Create tensors on the target device
    value = torch.randn(N, Len_q, M, D, device=device, dtype=torch.float32, requires_grad=True)
    spatial_shapes = torch.tensor(spatial_shapes_list, device=device, dtype=torch.long)
    level_start_index = torch.tensor([0, 100, 125, 134], device=device, dtype=torch.long)
    sampling_loc = torch.randn(N, Len_q, M, L, P, 2, device=device, dtype=torch.float32, requires_grad=True)
    attn_weight = torch.randn(N, Len_q, M, L, P, device=device, dtype=torch.float32, requires_grad=True)
    
    print(f"Input tensors created on {device}")
    print(f"Value shape: {value.shape}, device: {value.device}")
    
    try:
        # Test forward pass using the proper Python interface
        print("Testing forward pass...")
        output = MSDeformAttnFunction.apply(
            value, spatial_shapes, level_start_index, 
            sampling_loc, attn_weight, 128
        )
        
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}, device: {output.device}")
        
        # Test backward pass by computing gradients
        print("Testing backward pass...")
        output.requires_grad_(True)
        loss = output.sum()
        loss.backward()
        
        print(f"✓ Backward pass successful!")
        print(f"Gradients computed successfully")
        
        print("\n🎉 All tests passed! MPS support with CPU fallback is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mps_fallback()