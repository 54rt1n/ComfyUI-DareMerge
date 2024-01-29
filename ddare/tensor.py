# ddare/tensor.py
import torch
from typing import Optional, Literal

from .const import EPSILON, CHUNK_SIZE

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
    
    return sparsified_flat.view_as(model_a_param).to(dtype=model_a_param.dtype)

def relative_norm(tensor_a: torch.Tensor, tensor_b: torch.Tensor, eps : float = EPSILON) -> float:
    """
    Calculate the relative norm of two tensor tensors.

    Args:
        tensor_a (torch.Tensor): tensor tensor of this instance.
        tensor_b (torch.Tensor): tensor tensor of the other instance.
        eps (float): Epsilon value to avoid division by zero.

    Returns:
        float: Scaling factor.
    """
    norm_a = torch.norm(tensor_a)
    norm_b = torch.norm(tensor_b)
    return norm_b / (norm_a + eps)  # Adding epsilon to avoid division by zero

def l2_norm(tensor: torch.Tensor, eps : float = EPSILON) -> torch.Tensor:
    """
    Calculate the L2 norm of a tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to calculate the L2 norm for.
        eps (float): Epsilon value to avoid division by zero.
        
    Returns:
        float: The L2 norm of the tensor.
    """
    return tensor / (torch.norm(tensor) + eps)

def spectral_norm(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the spectral norm of a matrix.

    The spectral norm of a matrix is the largest singular value of the matrix. It's a measure
    of the matrix's ability to stretch or compress vectors when applied to them.

    Args:
        matrix (torch.Tensor): A 2D tensor representing the matrix for which the spectral norm
                               is to be computed. The tensor must have exactly two dimensions.

    Returns:
        torch.Tensor: A tensor containing a single scalar value, which is the spectral norm
                      of the input matrix.

    Raises:
        ValueError: If the input is not a 2D tensor.
    """
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D tensor.")

    # Compute the spectral norm (2-norm) using torch.linalg.norm
    norm = torch.linalg.norm(matrix, ord=2)

    return norm

def get_threshold_mask(model_a_param: torch.Tensor, model_b_param: torch.Tensor, device : torch.device, threshold: float, select: str, **kwargs) -> torch.Tensor:
    """
    Gets a mask of the delta parameter based on the specified sparsity level.
    
    Args:
        model_a_param (torch.Tensor): The parameter from model_a.
        model_b_param (torch.Tensor): The parameter from model_b.
        device (torch.device): The device to use for the mask.
        threshold (float): The sparsity level to use.
        invert (str): Whether to invert the mask or not.
        **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.
        
    Returns:
        torch.Tensor: The mask of the delta parameter.
    """

    model_a_flat = model_a_param.view(-1).float().to(device)
    model_b_flat = model_b_param.view(-1).float().to(device)
    delta_flat = model_b_flat - model_a_flat

    invertion = 0 if select == 'below' else 1
    if threshold == 1.0:
        mask = torch.zeros_like(delta_flat) == invertion
        #print(f"select: {select} threshold: {threshold} Selected: ({mask.sum()}) Excluded: ({(mask == False).sum()})")
    elif threshold == 0.0:
        mask = torch.ones_like(delta_flat) == invertion
        #print(f"select: {select} threshold: {threshold} Selected: ({mask.sum()}) Excluded: ({(mask == False).sum()})")
    else:
        absolute_delta = torch.abs(delta_flat)

        # We can easily overrun memory with large tensors, so we chunk the tensor
        delta_threshold = _threshold_in_chunks(tensor=absolute_delta, threshold=threshold, **kwargs)
        # Create a mask for values to keep or preserve (above the threshold)
        mask = absolute_delta < delta_threshold if select == 'below' else absolute_delta >= delta_threshold
        #print(f"select: {select} threshold: {threshold} Selected: ({mask.sum()}) Excluded: ({(mask == False).sum()}) Delta threshold: {delta_threshold} Mask: {absolute_delta.sum()} / {absolute_delta.numel()}")

    return mask.view_as(model_a_param)

def _threshold_in_chunks(tensor: torch.Tensor, threshold: float, threshold_type : str, **kwargs) -> float:
    """
    Processes the tensor in chunks and calculates the quantile thresholds for each chunk to determine our layer threshold.

    Args:
        tensor (torch.Tensor): The tensor to process.
        threshold (float): The quantile threshold to use.
        threshold_type (str): The type of threshold to use, either "median" or "quantile".
        
    Returns:
        float: The layer threshold.
    """
    thresholds = []
    for i in range(0, tensor.numel(), CHUNK_SIZE):
        chunk = tensor[i:i + CHUNK_SIZE]
        if chunk.numel() == 0:
            continue
        athreshold = torch.quantile(torch.abs(chunk), threshold).item()
        thresholds.append(athreshold)
    
    if threshold_type == "median":
        global_threshold = torch.median(torch.tensor(thresholds))
    else:
        sorted_thresholds = sorted(thresholds)
        index = int(threshold * len(sorted_thresholds))
        index = max(0, min(index, len(sorted_thresholds) - 1))
        global_threshold = sorted_thresholds[index]
    
    return global_threshold

def bernoulli_noise(layer : torch.Tensor, threshold : float) -> torch.Tensor:
    """
    Create a random mask of the same shape as the given layer.

    Args:
        layer (torch.Tensor): The layer to create a random mask for.
        threshold (float): Percentage of the layer to be true.

    Returns:
        torch.Tensor: A random mask of the same shape as the given layer.
        """
    return torch.rand(layer.shape) > threshold

def gaussian_noise(layer : torch.Tensor, mean : float = 0.5, std : float = 0.15):
    """
    Create a gaussian noise mask of the same shape as the given layer.
    
    Args:
        layer (torch.Tensor): The layer to create a gaussian noise mask for.
        mean (float): Mean of the gaussian distribution.
        std (float): Standard deviation of the gaussian distribution.
        
    Returns:
        torch.Tensor: A gaussian noise mask of the same shape as the given layer.
    """
    return torch.normal(mean, std, size=layer.shape) > mean

def divide_tensor_into_sets(tensor : torch.Tensor, n_sets : int):
    """
    Randomly divide a tensor into n different sets.

    Args:
        tensor (torch.Tensor): The tensor to be divided.
        n_sets (int): The number of sets to divide the tensor into.

    Returns:
        torch.Tensor: A tensor of the same shape as the input, where each element
                      indicates the set number (ranging from 0 to n_sets-1) to which
                      it has been assigned.
    """
    # Randomly assign each element to one of the n sets
    random_set_assignments = torch.randint(0, n_sets, tensor.shape, dtype=torch.int)

    return random_set_assignments