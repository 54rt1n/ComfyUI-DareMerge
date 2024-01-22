# components/mask_model.py
from comfy.model_patcher import ModelPatcher
import torch
from typing import Dict, Tuple

from ..ddare.util import cuda_memory_profiler, get_device
from ..ddare.mask import ModelMask
from ..ddare.const import MASK_CATEGORY


class MagnitudeMasker:
    """
    A managed state dict to allow for masking of model layers.  This is used for protecting
    layers from being overwritten by the merge process.
    """

    CHUNK_SIZE = 10**7  # Constant chunk size for memory management

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the masking process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_type": (["median", "quantile"], {"default": "median"}),
                "invert": (["No", "Yes"], {"default": "No"}),
            }
        }

    RETURN_TYPES = ("MODEL_MASK",)
    FUNCTION = "mask"
    CATEGORY = MASK_CATEGORY

    def mask(self, model_a: ModelPatcher, model_b: ModelPatcher, **kwargs) -> Tuple[ModelMask]:
        """
        Uses two ModelPatcher instances to determine the deltas between the two models and then create a mask for the deltas
        above and below a certain threshold.

        Args:
            model_a (ModelPatcher): The base model to be merged.
            model_b (ModelPatcher): The model to merge into the base model.
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the mask.
        """

        device = get_device()

        with cuda_memory_profiler():
            if len(model_a.patches) > 0:
                print("Model A has patches, applying them")
                model_a.patch_model(None, True)
                model_a_sd = model_a.model_state_dict()
                model_a.unpatch_model()
            else:
                model_a_sd = model_a.model_state_dict()

            if len(model_b.patches) > 0:
                print("Model B has patches, applying them")
                model_b.patch_model(None, True)
                model_b_sd = model_b.model_state_dict()
                model_b.unpatch_model()
            else:
                model_b_sd = model_b.model_state_dict()

            mm = ModelMask({})

            # Merge each parameter from model_b into model_a
            for k in model_a_sd.keys():
                if k not in model_b_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                a : torch.Tensor = model_a_sd[k]
                b : torch.Tensor = model_b_sd[k]

                layer_mask = self.get_threshold_mask(a, b, device=device, **kwargs)
                mm.add_layer_mask(k, layer_mask)

        return (mm,)

    def get_threshold_mask(self, model_a_param: torch.Tensor, model_b_param: torch.Tensor, device : torch.device, threshold: float, invert: str, **kwargs) -> torch.Tensor:
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

        invertion = 1 if invert == 'No' else 0
        if threshold == 1.0:
            mask = torch.ones_like(delta_flat) == invertion
        elif threshold == 0.0:
            mask = torch.zeros_like(delta_flat) == invertion
        else:
            absolute_delta = torch.abs(delta_flat)

            # We can easily overrun memory with large tensors, so we chunk the tensor
            delta_threshold = self.process_in_chunks(tensor=absolute_delta, threshold=threshold, **kwargs)
            # Create a mask for values to keep or preserve (above the threshold)
            mask = absolute_delta >= delta_threshold if invert == 'No' else absolute_delta < delta_threshold
            #print(f"Delta threshold: {delta_threshold} Mask: {absolute_delta.sum()} / {absolute_delta.numel()} invert: {invert} threshold: {threshold} Above: ({mask.sum()}) Below ({(mask == False).sum()})")

        return mask.view_as(model_a_param)

    def process_in_chunks(self, tensor: torch.Tensor, threshold: float, threshold_type : str, **kwargs) -> float:
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
        for i in range(0, tensor.numel(), self.CHUNK_SIZE):
            chunk = tensor[i:i + self.CHUNK_SIZE]
            if chunk.numel() == 0:
                continue
            threshold = torch.quantile(torch.abs(chunk), threshold).item()
            thresholds.append(threshold)
        
        if threshold_type == "median":
            global_threshold = torch.median(torch.tensor(thresholds))
        else:
            sorted_thresholds = sorted(thresholds)
            index = int(threshold * len(sorted_thresholds))
            index = max(0, min(index, len(sorted_thresholds) - 1))
            global_threshold = sorted_thresholds[index]
        
        return global_threshold

