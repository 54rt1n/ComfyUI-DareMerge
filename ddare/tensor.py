# ddare/tensor.py

import torch
from typing import Optional, Literal

from .const import EPSILON

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

def relative_norm(weight_a: torch.Tensor, weight_b: torch.Tensor, eps : float = EPSILON) -> float:
    """
    Calculate the relative norm of two weight tensors.

    Args:
        weight_a (torch.Tensor): Weight tensor of this instance.
        weight_b (torch.Tensor): Weight tensor of the other instance.

    Returns:
        float: Scaling factor.
    """
    norm_a = torch.norm(weight_a)
    norm_b = torch.norm(weight_b)
    return norm_b / (norm_a + eps)  # Adding epsilon to avoid division by zero
