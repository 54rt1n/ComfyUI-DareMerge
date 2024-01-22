# ddare/merge.py
# Credit to https://github.com/Gryphe/MergeMonster
import torch
from typing import Optional, Literal

from .const import EPSILON

def merge_tensors(method: str, v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    if method == "lerp":
        return merge_tensors_lerp(v0, v1, t)
    elif method == "slerp":
        return merge_tensors_slerp(v0, v1, t)
    elif method == "slice":
        return merge_tensors_slice(v0, v1, t)
    elif method == "cyclic":
        return merge_tensors_cyclic(v0, v1, t)
    elif method == "gradient":
        return merge_tensors_gradient(v0, v1, t)

def merge_tensors_lerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    
    result = ((1 - t) * v0) + (t * v1)
    
    return result

def merge_tensors_slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, dot_threshold: float = 0.9995, eps: float = EPSILON) -> torch.Tensor:
    """Spherical linear interpolation between two tensors or linear interpolation if they are one-dimensional.
       Full credit to https://github.com/cg123/mergekit for the original code."""

    # We LERP single dimensional tensors
    if v0.dim() == 1 and v1.dim() == 1:
        return merge_tensors_lerp(v0, v1, t)

    # Make copies of the original tensors to use for interpolation
    v0_copy = v0.clone()
    v1_copy = v1.clone()

    # Normalize the original tensors for angle computation
    v0 = safe_normalize(v0, eps)
    v1 = safe_normalize(v1, eps)

    # Compute the cosine of the angle between the normalized vectors.
    dot = (v0 * v1).sum()

    # If the inputs are too close, linearly interpolate using the original tensors.
    if abs(dot) > dot_threshold:
        return merge_tensors_lerp(v0_copy, v1_copy, t)

    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    # Use the weights with the original tensors (not normalized) for the final result
    result = s0 * v0_copy + s1 * v1_copy

    return result

# MODEL 1 > 10% blend > MODEL 2
def merge_tensors_slice(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    # We're only working on the second dimension here
    if v0.dim() == 2:
        # Calculate the slice indices for each tensor
        slice_index_0 = int(v0.shape[1] * (1 - t))
        slice_index_1 = v1.shape[1] - slice_index_0
    
        blend_slice_size = int(v0.shape[1] * 0.05)
        blend_slice_0 = v0.narrow(1, slice_index_0 - blend_slice_size, blend_slice_size * 2)
        blend_slice_1 = v1.narrow(1, slice_index_0 - blend_slice_size, blend_slice_size * 2)
        blended_slice = blend_slice_0

        # Apply gradient blending
        for i in range(blend_slice_size * 2):
            blend_ratio = i / (blend_slice_size * 2)
            blended_slice[:, i] = (blend_slice_1[:, i] * blend_ratio) + (blend_slice_0[:, i] * (1 - blend_ratio))
    
        slice_index_0 = slice_index_0 - blend_slice_size
        slice_index_1 = slice_index_0 + blend_slice_size + blend_slice_size
    
        # Perform slicing
        slice_0 = v0.narrow(1, 0, slice_index_0)
        slice_1 = v1.narrow(1, slice_index_1, v1.shape[1] - slice_index_1)
    
        # Concatenate the slices
        result = torch.cat([slice_0, blended_slice, slice_1], dim=1)
    
        return result
    else:
        return v0

# MODEL 1 > 10% blend > 10% of MODEL 2 > 10% blend > MODEL 1, with varying starting positions as defined by t
def merge_tensors_cyclic(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    # We're only working on the second dimension here
    if v0.dim() == 2:
        blend_slice_size = int(v0.shape[1] * 0.05) # Blending zone is eventually multiplied by two due to overlap
        v1_slice_size = int(v0.shape[1] * 0.10) # 10% of Model 2, accounting for the 5% blend zone on both sides. So kinda 15%.

        slice_index_0 = int(v0.shape[1] * (1 - t)) - blend_slice_size # Model 1, first slice length

        # First MODEL 1 > MODEL 2 blend
        # -----------------------
        blend_slice_0_0 = v0.narrow(1, slice_index_0, blend_slice_size * 2)
        blend_slice_0_1 = v1.narrow(1, slice_index_0, blend_slice_size * 2)
        blended_slice_0 = blend_slice_0_0

        # Apply gradient blending
        for i in range(blend_slice_size * 2):
            blend_ratio = i / (blend_slice_size * 2)
            blended_slice_0[:, i] = (blend_slice_0_0[:, i] * (1 - blend_ratio)) + (blend_slice_0_1[:, i] * blend_ratio)

        # Second MODEL 2 > MODEL 1 blend
        # -----------------------
        blend_slice_1_0 = v0.narrow(1, slice_index_0 + (blend_slice_size * 2) + v1_slice_size, blend_slice_size * 2)
        blend_slice_1_1 = v1.narrow(1, slice_index_0 + (blend_slice_size * 2) + v1_slice_size, blend_slice_size * 2)
        blended_slice_1 = blend_slice_1_0

        # Apply gradient blending
        for i in range(blend_slice_size * 2):
            blend_ratio = i / (blend_slice_size * 2)
            blended_slice_1[:, i] = (blend_slice_1_1[:, i] * (1 - blend_ratio)) + (blend_slice_1_0[:, i] * blend_ratio)

        # Time to out main candidates into various pieces
        m1len_0   = slice_index_0
        m2start   = slice_index_0 + (blend_slice_size * 2)
        m1start_1 = m2start + v1_slice_size + (blend_slice_size * 2)
        m2end_1   = v1.shape[1] - m1start_1

        # print(f"M1 0-{m1len_0} > B1 {m1len_0}-{m1len_0+(blend_slice_size * 2)} > M2 {m2start}-{m2start+v1_slice_size} > B2 {m2start+v1_slice_size}-{m1start_1} > M1 {m1start_1}-{v1.shape[1]}")
        
        slice_0_0 = v0.narrow(1, 0, m1len_0) # Model 1, first piece
        slice_1_0 = v1.narrow(1, m2start, v1_slice_size) # Model 2 slice
        slice_0_1 = v0.narrow(1, m1start_1, m2end_1) # Model 1, second piece
    
        # Concatenate the slices
        result = torch.cat([slice_0_0, blended_slice_0, slice_1_0, blended_slice_1, slice_0_1], dim=1)
    
        return result
    else:
        return v0

# Model 1 > Model 2 > Model 1, with t defining the peak of the gradient along the tensor's width
def merge_tensors_gradient(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    device = v0.device
    if v0.dim() == 2:
        total_length = v0.shape[1]
        peak = int(total_length * (1 - t))

        # Create an index array
        indices = torch.arange(total_length).float()

        # Vectorized computation of blend ratios
        blend_ratios = torch.zeros_like(indices)
        blend_ratios[:peak] = (indices[:peak] / peak) * 0.9  # Scale to max 0.9 for v1
        blend_ratios[peak:] = torch.flip(indices[:total_length - peak], dims=[0]) / (total_length - peak) * 0.9  # Scale to max 0.9 for v1

        # Ensure that v0 still has influence
        v0_ratios = 1 - blend_ratios

        # Vectorized blending of the tensors
        result = (v1 * blend_ratios.unsqueeze(0).to(device)) + (v0 * v0_ratios.unsqueeze(0).to(device))

        return result
    else:
        return v0

def safe_normalize(tensor: torch.Tensor, eps: float = EPSILON):
    norm = tensor.norm()
    if norm > eps:
        return tensor / norm
    return tensor

def get_ties_mask(delta: torch.Tensor, method: Literal["sum", "count"] = "sum", mask_dtype: Optional[torch.dtype] = None, **kwargs) -> torch.Tensor:
    """
        TIES-merging https://arxiv.org/abs/2306.01708 uses sign agreement, protecting from
        major perturbations in the opposite direction of the base model
    
        Returns a mask determining which delta vectors should be merged
        into the final model.

        For the methodology described in the paper use 'sum'. For a
        simpler naive count of signs, use 'count'.
    """
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = (sign * delta.abs()).sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign

def dare_ties_sparsification(model_a_param: torch.Tensor, model_b_param: torch.Tensor,
                             drop_rate: float, ties : str, rescale : str, device : torch.device,
                             **kwargs) -> torch.Tensor:
    """
    DARE-TIES sparsification uses a stochastic mask to determine which deltas to apply
    and then sign-agreement to determine which deltas to merge into the final model.

    Args:
        model_a_param (torch.Tensor): The base model parameter tensor.
        model_b_param (torch.Tensor): The model parameter tensor to merge into the base model.
        drop_rate (float): The drop rate for the stochastic mask.
        ties (str): Whether to use the TIES-merging method.
        rescale (str): Whether to rescale the remaining deltas.
        device (torch.device): The device to use for the merge.
        
    Returns:
        torch.Tensor: The updated parameter tensor.
    """

    model_a_flat = model_a_param.view(-1).float().to(device)
    model_b_flat = model_b_param.view(-1).float().to(device)
    delta_flat = model_b_flat - model_a_flat

    dare_mask = torch.bernoulli(torch.full(delta_flat.shape, 1 - drop_rate, device=device)).bool()
    # The paper says we should rescale, but it yields terrible results for SD.
    if rescale == "on":
        # Rescale the remaining deltas
        delta_flat = delta_flat / (1 - drop_rate)
    
    if ties != "off":
        ties_mask = get_ties_mask(delta_flat, ties)
        dare_mask = dare_mask & ties_mask
        del ties_mask

    sparsified_flat = torch.where(dare_mask, model_a_flat + delta_flat, model_a_flat)
    del delta_flat, model_a_flat, model_b_flat, dare_mask
    
    return sparsified_flat.view_as(model_a_param)
