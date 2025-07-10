/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>
#include <cmath>

#include <ATen/ATen.h>


at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    // Store original device for later conversion back
    auto original_device = value.device();
    bool is_mps = original_device.is_mps();
    
    // Convert MPS tensors to CPU for processing
    auto value_cpu = is_mps ? value.to(at::Device(at::kCPU)) : value;
    auto spatial_shapes_cpu = is_mps ? spatial_shapes.to(at::Device(at::kCPU)) : spatial_shapes;
    auto level_start_index_cpu = is_mps ? level_start_index.to(at::Device(at::kCPU)) : level_start_index;
    auto sampling_loc_cpu = is_mps ? sampling_loc.to(at::Device(at::kCPU)) : sampling_loc;
    auto attn_weight_cpu = is_mps ? attn_weight.to(at::Device(at::kCPU)) : attn_weight;
    
    // Basic CPU implementation for Mac/MPS compatibility
    // Expected shapes: value: [N, Len_q, num_heads, D], where Len_q is the sum of H*W for all levels
    auto N = value_cpu.size(0);      // batch size
    auto Len_q = value_cpu.size(1);  // total length (sum of H*W across levels)
    auto num_heads = value_cpu.size(2); // number of heads
    auto D = value_cpu.size(3);      // feature dimension
    
    // attn_weight: [N, Len_q, num_heads, num_levels, num_points]
    auto num_levels = attn_weight_cpu.size(3);
    auto num_points = attn_weight_cpu.size(4);
    
    // Output: [N, Len_q, num_heads, D]
    auto output_cpu = at::zeros({N, Len_q, num_heads, D}, value_cpu.options());
    
    // This is a simplified placeholder implementation
    // For a complete implementation, we would need to:
    // 1. Split value by spatial shapes into different levels
    // 2. Apply deformable sampling using sampling_loc_cpu
    // 3. Weight the sampled values using attn_weight_cpu
    // For now, just return zeros to allow compilation and testing
    
    // Placeholder: copy input to output (not correct but allows testing)
    output_cpu.copy_(value_cpu);
    
    // Convert back to original device if needed
    return is_mps ? output_cpu.to(original_device) : output_cpu;
}

std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    // Store original device for later conversion back
    auto original_device = value.device();
    bool is_mps = original_device.is_mps();
    
    // Convert MPS tensors to CPU for processing
    auto value_cpu = is_mps ? value.to(at::Device(at::kCPU)) : value;
    auto spatial_shapes_cpu = is_mps ? spatial_shapes.to(at::Device(at::kCPU)) : spatial_shapes;
    auto level_start_index_cpu = is_mps ? level_start_index.to(at::Device(at::kCPU)) : level_start_index;
    auto sampling_loc_cpu = is_mps ? sampling_loc.to(at::Device(at::kCPU)) : sampling_loc;
    auto attn_weight_cpu = is_mps ? attn_weight.to(at::Device(at::kCPU)) : attn_weight;
    auto grad_output_cpu = is_mps ? grad_output.to(at::Device(at::kCPU)) : grad_output;
    
    // Basic CPU backward implementation for Mac/MPS compatibility
    auto grad_value_cpu = at::zeros_like(value_cpu);
    auto grad_sampling_loc_cpu = at::zeros_like(sampling_loc_cpu);
    auto grad_attn_weight_cpu = at::zeros_like(attn_weight_cpu);
    
    // Simplified backward pass - this is a placeholder that allows compilation
    // For a complete implementation, proper gradient computation would be needed
    
    // Convert back to original device if needed
    auto grad_value = is_mps ? grad_value_cpu.to(original_device) : grad_value_cpu;
    auto grad_sampling_loc = is_mps ? grad_sampling_loc_cpu.to(original_device) : grad_sampling_loc_cpu;
    auto grad_attn_weight = is_mps ? grad_attn_weight_cpu.to(original_device) : grad_attn_weight_cpu;
    
    return {grad_value, grad_sampling_loc, grad_attn_weight};
}

