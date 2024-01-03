# merge/dare.py
import torch
from typing import Dict, Tuple, Optional

from comfy.model_patcher import ModelPatcher

from .mergeutil import merge_tensors


class DareModelMerger:
    """
    A class to merge two diffusion U-Net models using calculated deltas, sparsification,
    and a weighted consensus method. This is the DARE method.
    
    https://arxiv.org/pdf/2311.03099.pdf
    """

    CHUNK_SIZE = 10**7  # Constant chunk size for memory management

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the merging process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "optional": {
                "base_model": ("MODEL",),
            },
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "drop_rate": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42}),
                "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["comfy", "lerp", "slerp", "gradient"], ),
                "exclude_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "include_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_type": (["median", "quantile"], {"default": "median"}),
                "invert": (["No", "Yes"], {"default": "No"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "ddare/dare"

    def merge(self, model_a: ModelPatcher, model_b: ModelPatcher, 
              input: float, middle: float, out: float, time: float, method : str,
              seed : Optional[int] = None, clear_cache : bool = True,
              base_model: Optional[ModelPatcher] = None,
              **kwargs) -> Tuple[ModelPatcher]:
        """
        Merges two ModelPatcher instances based on the weighted consensus of their parameters and sparsity.

        Args:
            model1 (ModelPatcher): The base model to be merged.
            model2 (ModelPatcher): The model to merge into the base model.
            input (float): The ratio (lambda) of the input layer to keep from model1.
            middle (float): The ratio (lambda) of the middle layers to keep from model1.
            out (float): The ratio (lambda) of the output layer to keep from model1.
            method (str): The method to use for merging, either "lerp", "slerp", or "gradient".
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the merged ModelPatcher instance.
        """

        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        m = model_a.clone()  # Clone model1 to keep its structure
        model_a_sd = m.model_state_dict()  # State dict of model1
        if base_model is not None:
            model_base_sd = base_model.model_state_dict()  # State dict of base model
        else:
            model_base_sd = None
        kp = model_b.get_key_patches("diffusion_model.")  # Get the key patches from model2

        # Merge each parameter from model2 into model1
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
            base : torch.Tensor = model_base_sd[k] if model_base_sd is not None else None
            a : torch.Tensor = model_a_sd[k]
            b : torch.Tensor = kp[k][-1]

            # Debugging
            # our 'Tensor's might be a tuple sometimes if it's part of a chain.  This logic is very hacky and could be flawed.
            # typer = lambda x: type(x) if not isinstance(x, tuple) else [typer(y) for y in x]

            if isinstance(base, tuple):
                #print('chain', a[0], a[-1], len(a), typer(a))
                base = self.patcher(base_model, k)
                if base is None:
                    continue
            elif base is not None:
                base = base.copy_(base)
            
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

            #print(f"Processing Layer {k} with ratio {ratio}")
            sparsified_delta = self.apply_sparsification(base, a, b, device=device, **kwargs)
            if method == "comfy":
                merged_layer = sparsified_delta

                strength_patch = 1.0 - ratio
                strength_model = ratio
            else:
                merged_layer = merge_tensors(method, a.to(device), sparsified_delta.to(device), 1 - ratio)

                strength_model = 0
                strength_patch = 1.0

            del base, a, b
            
            # Apply the sparsified delta as a patch
            nv = (merged_layer.to('cpu'),)

            m.add_patches({k: nv}, strength_patch, strength_model)

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (m,)

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

    def process_in_chunks(self, tensor: torch.Tensor, sparsity: float, threshold_type : str, **kwargs) -> torch.Tensor:
        """
        Processes the tensor in chunks and calculates the quantile thresholds for each chunk.
        """
        thresholds = []
        for i in range(0, tensor.numel(), self.CHUNK_SIZE):
            chunk = tensor[i:i + self.CHUNK_SIZE]
            if chunk.numel() == 0:
                continue
            threshold = torch.quantile(torch.abs(chunk), sparsity).item()
            thresholds.append(threshold)
        
        if threshold_type == "median":
            global_threshold = torch.median(torch.tensor(thresholds))
        else:
            sorted_thresholds = sorted(thresholds)
            index = int(sparsity * len(sorted_thresholds))
            index = max(0, min(index, len(sorted_thresholds) - 1))
            global_threshold = sorted_thresholds[index]
        
        return global_threshold

    def get_threshold_mask(self, delta_param: torch.Tensor, sparsity: float, invert: str, **kwargs) -> torch.Tensor:
        """
        Gets a mask of the delta parameter based on the specified sparsity level.
        
        Args:
            delta_param (torch.Tensor): The delta parameter tensor.
            sparsity (float): The fraction of elements to set to zero.  0 = include all, 1 = exclude all.
            invert (str): Whether to invert the sparsification, i.e., keep the least significant changes.
            
        Returns:
            torch.Tensor: The mask of the delta parameter.
        """

        invertion = 1 if invert == 'No' else 0
        if sparsity == 1.0:
            return torch.ones_like(delta_param) == invertion
        elif sparsity == 0.0:
            return torch.zeros_like(delta_param) == invertion

        absolute_delta = torch.abs(delta_param)

        # We can easily overrun memory with large tensors, so we chunk the tensor
        delta_threshold = self.process_in_chunks(tensor=absolute_delta, sparsity=sparsity, **kwargs)
        print(f"Delta threshold: {delta_threshold} Mask: {absolute_delta.sum()} / {absolute_delta.numel()} invert: {invert} sparsity: {sparsity}")

        # Create a mask for values to keep or preserve (above the threshold)
        mask = absolute_delta >= delta_threshold if invert == 'No' else absolute_delta < delta_threshold
        return mask

    def apply_sparsification(self, base_model_param: Optional[torch.Tensor], model_a_param: torch.Tensor, model_b_param: torch.Tensor,
                             exclude_a: float, include_b: float, invert : str, drop_rate: float, device : torch.device, **kwargs) -> torch.Tensor:
        """
        Applies sparsification to a tensor based on the specified sparsity level.
        """

        model_a_flat = model_a_param.view(-1).float().to(device)
        model_b_flat = model_b_param.view(-1).float().to(device)
        delta_flat = model_b_flat - model_a_flat

        if base_model_param is not None:
            base_model_flat = base_model_param.view(-1).float().to(device)
            delta_a_flat = model_a_flat - base_model_flat
            delta_b_flat = model_b_flat - base_model_flat
            
            include_mask = self.get_threshold_mask(delta_b_flat, include_b, invert, **kwargs)
            exclude_mask = self.get_threshold_mask(delta_a_flat, exclude_a, invert, **kwargs)
            del base_model_flat, delta_a_flat, delta_b_flat
        else:
            include_mask = torch.ones_like(model_a_flat).bool()
            exclude_mask = torch.ones_like(model_a_flat).bool()
            
        base_mask = include_mask & (~exclude_mask)
        
        dare_mask = torch.bernoulli(torch.full(delta_flat.shape, 1 - drop_rate, device=device)).bool()
        # The paper says we should rescale, but it yields terrible results
        # delta_flat = delta_flat / (1 - drop_rate)  # Rescale the remaining deltas
        
        mask = dare_mask & base_mask
        print(f"mask nonzero count: {torch.count_nonzero(mask)} dare nonzero count: {torch.count_nonzero(dare_mask)} base nonzero count: {torch.count_nonzero(base_mask)} include nonzero count: {torch.count_nonzero(include_mask)} exclude nonzero count: {torch.count_nonzero(exclude_mask)}")

        sparsified_flat = torch.where(mask, model_a_flat + delta_flat, model_a_flat)
        del mask, delta_flat, include_mask, exclude_mask, base_mask, model_a_flat, model_b_flat
        
        return sparsified_flat.view_as(model_a_param)
