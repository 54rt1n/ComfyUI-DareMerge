# ddare/merge.py
# Credit to https://github.com/Gryphe/MergeMonster
# and https://github.com/WASasquatch/FreeU_Advanced/nodes.py
import torch
from typing import Optional

from .const import EPSILON

METHODS = ["lerp", "slerp", "slice", "cyclic", "gradient", "hslerp", "bislerp", "colorize", "cosine", "cubic", "scaled_add"]

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
    elif method == "hslerp":
        return merge_tensors_hslerp(v0, v1, t)
    elif method == "bislerp":
        return merge_tensors_bislerp(v0, v1, t)
    elif method == "colorize":
        return merge_tensors_colorize(v0, v1, t)
    elif method == "cosine":
        return merge_tensors_cosine_interpolation(v0, v1, t)
    elif method == "cubic":
        return merge_tensors_cubic_interpolation(v0, v1, t)
    elif method == "scaled_add":
        return merge_tensors_scaled_add(v0, v1, t)
    else:
        raise ValueError(f"Unknown merge method: {method}")

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

def normalize_rescale(latent: torch.Tensor, target_min : Optional[float] = None, target_max : Optional[float] = None) -> torch.Tensor:
    """
    Normalize a tensor `latent` between `target_min` and `target_max`.

    Args:
        latent (torch.Tensor): The input tensor to be normalized.
        target_min (float, optional): The minimum value after normalization.
            - When `None` min will be tensor min range value.
        target_max (float, optional): The maximum value after normalization.
            - When `None` max will be tensor max range value.

    Returns:
        torch.Tensor: The normalized tensor
    """
    min_val = latent.min()
    max_val = latent.max()

    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val

    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled

def merge_tensors_hslerp(a : torch.Tensor, b : torch.Tensor, t : float) -> torch.Tensor:
    """
    Perform Hybrid Spherical Linear Interpolation (HSLERP) between two tensors.

    This function combines two input tensors `a` and `b` using HSLERP, which is a specialized
    interpolation method for smooth transitions between orientations or colors.

    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.

    Returns:
        tensor: The result of HSLERP interpolation between `a` and `b`.

    Note:
        HSLERP provides smooth transitions between orientations or colors, particularly useful
        in applications like image processing and 3D graphics.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    num_channels = a.size(1)

    interpolation_tensor = torch.zeros(1, num_channels, 1, 1, device=a.device, dtype=a.dtype)
    interpolation_tensor[0, 0, 0, 0] = 1.0

    result = (1 - t) * a + t * b

    if t < 0.5:
        result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    else:
        result -= (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor

    return result

def merge_tensors_bislerp(a : torch.Tensor, b : torch.Tensor, t : float) -> torch.Tensor:
    """
    Perform Bidirectional Spherical Linear Interpolation (BISLERP) between two tensors.
    
    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.
        
    Returns:
        tensor: The result of BISLERP interpolation between `a` and `b`.
    """

    return normalize_rescale((1 - t) * a + t * b)

def merge_tensors_colorize(a : torch.Tensor, b : torch.Tensor, t : float) -> torch.Tensor:
    """
    Perform colorization between two tensors.
    
    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.
        
    Returns:
        tensor: The result of colorization between `a` and `b`.
    """
    
    return a + (b - a) * t

def merge_tensors_cosine_interpolation(a : torch.Tensor, b : torch.Tensor, t : float) -> torch.Tensor:
    """
    Perform cosine interpolation between two tensors.
    
    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.
        
    Returns:
        tensor: The result of cosine interpolation between `a` and `b`.
    """
    
    return (a + b - (a - b) * torch.cos(t * torch.tensor(torch.pi))) / 2

def merge_tensors_cubic_interpolation(a : torch.Tensor, b : torch.Tensor, t : float) -> torch.Tensor:
    """
    Perform cubic interpolation between two tensors.
    
    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.
        
    Returns:
        tensor: The result of cubic interpolation between `a` and `b`.
    """

    return a + (b - a) * (3 * t ** 2 - 2 * t ** 3)
    
def merge_tensors_scaled_add(a : torch.Tensor, b : torch.Tensor, t : float) -> torch.Tensor:
    """
    Perform additive blending between two tensors.  This adds the second tensor, scaled by `t`, to the first tensor.
    
    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.
        
    Returns:
        tensor: The result of additive blending between `a` and `b`. 
    """
    
    return (a + b * t) / (1 + t)

def merge_tensors_slice(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Blend two tensors by slicing them and blending the slices.
    
    Args:
        v0 (tensor): The first input tensor.
        v1 (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.
        
    Returns:
        tensor: The result of blending between `v0` and `v1`.
    """
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