# merge/mag.py
import torch
from typing import Dict, Tuple, Optional

from comfy.model_patcher import ModelPatcher

from .mergeutil import merge_tensors

class MagnitudePruningModelMerger:
    """
    A class to merge two diffusion U-Net models using calculated deltas, sparsification,
    and a weighted consensus method.  This is the Magnitude Pruning method.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the merging process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["lerp", "slerp", "gradient"], ),
                "sparsity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_type": (["median", "quantile"], ),
                "invert": (["No", "Yes"], ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "ddare/magnitude_pruning"

    def apply_sparsification(self, base_param: torch.Tensor, target_param: torch.Tensor, sparsity: float,
                             threshold_type : str, invert : str, device : torch.device, clear_cache : bool = False,
                             **kwargs) -> torch.Tensor:
        """
        Applies sparsification to a tensor based on the specified sparsity level, with chunking for large tensors.

        Args:
            base_param (torch.Tensor): The corresponding parameter from the base model.
            target_param (torch.Tensor): The corresponding parameter from the update model.
            sparsity (float): The fraction of elements to set to zero.
            threshold_type (str): The type of threshold to use, either "median" or "quantile".
            invert (str): Whether to invert the sparsification, i.e., keep the least significant changes.
            clear_cache (bool): Whether to clear the CUDA cache after each chunk. Default is False.

        Returns:
            torch.Tensor: The tensor with insignificant changes replaced by the base model's values.
        """
        # Ensure the delta and base_param are float tensors for quantile calculation, and on the right device
        target_param = target_param.to(device)
        base_param = base_param.to(device)
        delta = target_param - base_param
        base_param_flat = base_param.view(-1).float()
        delta_flat = delta.view(-1).float().to(device)
        absolute_delta = torch.abs(delta_flat)

        # We can easily overrun memory with large tensors, so we chunk the tensor
        # Define chunk size and prepare to collect thresholds
        chunk_size = 10**7
        thresholds = []

        # Process each chunk to determine thresholds
        for i in range(0, absolute_delta.numel(), chunk_size):
            chunk = absolute_delta[i:i + chunk_size]
            if chunk.numel() == 0:
                continue
            k = int(sparsity * chunk.numel())
            if k > 0:
                threshold = torch.quantile(chunk, sparsity)
            else:
                threshold = torch.tensor(0.0)
            thresholds.append(threshold)

        # Determine a global threshold
        
        if threshold_type == "median":
            global_threshold = torch.median(torch.tensor(thresholds))
        else:
            sorted_thresholds = sorted(thresholds)
            index = int(sparsity * len(sorted_thresholds))
            index = max(0, min(index, len(sorted_thresholds) - 1))
            global_threshold = sorted_thresholds[index]

        # Create a mask for values to keep (above the threshold)
        mask = absolute_delta >= global_threshold if invert == 'No' else absolute_delta < global_threshold
        print(f"Global threshold: {global_threshold} Mask: {mask.abs().sum()} / {mask.numel()}")

        # Apply the mask to the delta, replace other values with the base model's parameters
        sparsified_flat = torch.where(mask, base_param_flat, base_param_flat + delta_flat)
        del mask, absolute_delta, delta_flat, base_param_flat, global_threshold, thresholds
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return sparsified_flat.view_as(base_param)

    def patcher(self, model: ModelPatcher, key : str) -> Optional[torch.Tensor]:
        # This is slow, but seems to work
        model_sd = model.model_state_dict()
        if key not in model_sd:
            print("could not patch. key doesn't exist in model:", key)
            return None

        weight : torch.Tensor = model_sd[key]

        temp_weight = weight.to(torch.float32, copy=True)
        out_weight = model.calculate_weight(model.patches[key], temp_weight, key).to(weight.dtype)
        return out_weight

    def merge(self, model_a: ModelPatcher, model_b: ModelPatcher,
              input : float, middle : float, out : float, time : float, method : str,
              **kwargs) -> Tuple[ModelPatcher]:
        """
        Merges two ModelPatcher instances based on the weighted consensus of their parameters and sparsity.

        Args:
            model_a (ModelPatcher): The base model to be merged.
            model_b (ModelPatcher): The model to merge into the base model.
            input (float): The ratio (lambda) of the input layer to keep from model_a.
            middle (float): The ratio (lambda) of the middle layers to keep from model_a.
            out (float): The ratio (lambda) of the output layer to keep from model_a.
            time (float): The ratio (lambda) of the time layers to keep from model_a.
            method (str): The method to use for merging, either "lerp", "slerp", or "gradient".
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the merged ModelPatcher instance.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = model_a.clone()  # Clone model_a to keep its structure
        model_a_sd = m.model_state_dict()  # State dict of model_a
        kp = model_b.get_key_patches("diffusion_model.")  # Get the key patches from model_b

        # Merge each parameter from model_b into model_a
        for k in kp:
            if k not in model_a_sd:
                print("could not patch. key doesn't exist in model:", k)
                continue

            k_unet = k[len("diffusion_model."):]

            # Get our ratio for this layer
            if k_unet.startswith("input"):
                ratio = input
            elif k_unet.startswith("middle"):
                ratio = middle
            elif k_unet.startswith("out"):
                ratio = out
            elif k_unet.startswith("time"):
                ratio = time
            else:
                print(f"Unknown key: {k}, skipping.")
                continue

            # Apply sparsification by the delta, I don't know if all of this cuda stuff is necessary
            # but I had so many memory issues that I'm being very careful
            a : torch.Tensor = model_a_sd[k]
            b : torch.Tensor = kp[k][-1]

            # Debugging
            # our 'Tensor's might be a tuple sometimes if it's part of a chain.  This logic is very hacky and could be flawed.
            # typer = lambda x: type(x) if not isinstance(x, tuple) else [typer(y) for y in x]
            
            if isinstance(a, tuple):
                #print('chain', a[0], a[-1], len(a), typer(a))
                a = self.patcher(model_a, k)
                if a is None:
                    continue
            else:
                a = a.copy_(a)
            
            if isinstance(b, tuple):
                #print('chain', b[0], b[-1], len(b), typer(b))
                b = self.patcher(model_b, k)
                if b is None:
                    continue
            else:
                b = b.copy_(b)

            sparsified_delta = self.apply_sparsification(a, b, device=device, **kwargs)
            #merged_layer = merge_tensors(method, a.to(device), sparsified_delta.to(device), 1 - ratio)
            merged_layer = sparsified_delta.to('cpu')

            nv = (merged_layer,)

            del a, b
            
            # Apply the sparsified delta as a patch
            print(f"Patching {k} with {nv[0].shape} {1 - ratio}")
            m.add_patches({k: nv}, 1 - ratio, ratio)

        return (m,)

