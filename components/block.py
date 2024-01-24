# components/block.py
from comfy.model_patcher import ModelPatcher
import torch
from typing import Dict, Tuple, Optional

from ..ddare.const import UNET_CATEGORY
from ..ddare.mask import ModelMask
from ..ddare.merge import merge_tensors
from ..ddare.util import cuda_memory_profiler, get_device, get_patched_state


class BlockUnetMerger:
    """
    A class to merge two diffusion U-Net models using m mask.
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
                "model_mask": ("MODEL_MASK",),
                "time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["comfy", "lerp", "slerp", "gradient"], ),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = UNET_CATEGORY

    def merge(self, model_a: ModelPatcher, model_b: ModelPatcher,
              input : float, middle : float, output : float, out : float, time : float, method : str,
              clear_cache : bool = True, model_mask: Optional[ModelMask] = None,
              **kwargs) -> Tuple[ModelPatcher]:
        """
        Merges two ModelPatcher instances based on the weighted consensus of their parameters and sparsity.

        Args:
            model_a (ModelPatcher): The base model to be merged.
            model_b (ModelPatcher): The model to merge into the base model.
            input (float): The ratio (lambda) of the input layer to keep from model_a.
            middle (float): The ratio (lambda) of the middle layers to keep from model_a.
            output (float): The ratio (lambda) of the output layer to keep from model_a.
            out (float): The ratio (lambda) of the output layer to keep from model_a.
            time (float): The ratio (lambda) of the time layers to keep from model_a.
            method (str): The method to use for merging, either "comfy", "lerp", "slerp", or "gradient".
            clear_cache (bool): Whether to clear the CUDA cache after each chunk. Default is True.
            model_mask (ModelMask): A ModelMask instance to use for masking the model. Default is None.
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the merged ModelPatcher instance.
        """

        device = get_device()
        
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        m = model_a.clone()  # Clone model_a to keep its structure

        with cuda_memory_profiler():
            model_a_sd = get_patched_state(m)
            model_b_sd = get_patched_state(model_b)

            # Merge each parameter from model_b into model_a
            for k in model_a_sd.keys():
                if k not in model_b_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                k_unet = k[len("diffusion_model."):]

                # Get our ratio for this layer
                if k_unet.startswith("input"):
                    ratio = input
                elif k_unet.startswith("middle"):
                    ratio = middle
                elif k_unet.startswith("output"):
                    ratio = output
                elif k_unet.startswith("out"):
                    ratio = out
                elif k_unet.startswith("time"):
                    ratio = time
                else:
                    print(f"Unknown key: {k}, skipping.")
                    continue

                # Apply sparsification by the delta for this layer
                mask : torch.Tensor = model_mask.get_layer_mask(k) if model_mask is not None else None
                a : torch.Tensor = model_a_sd[k]
                b : torch.Tensor = model_b_sd[k]
                if mask is None:
                    mask = torch.ones_like(a)

                result_tensor = torch.where(mask.to(device), a.to(device), b.to(device))
                del mask

                if method == "comfy":
                    strength_patch = 1.0 - ratio
                    strength_model = ratio
                else:
                    result_tensor = merge_tensors(method, a.to(device), result_tensor, 1 - ratio)

                    strength_model = 0
                    strength_patch = 1.0

                m.add_patches({k: (result_tensor.to('cpu'),)}, strength_patch, strength_model)

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (m,)