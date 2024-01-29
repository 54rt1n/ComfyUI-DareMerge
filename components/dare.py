# components/dare.py
from comfy.model_patcher import ModelPatcher
import torch
from typing import Dict, Tuple, Optional

from ..ddare.const import UNET_CATEGORY, LAYER_GRADIENT, MODEL_MASK
from ..ddare.mask import ModelMask
from ..ddare.merge import merge_tensors, METHODS
from ..ddare.tensor import dare_ties_sparsification
from ..ddare.util import cuda_memory_profiler, get_device, get_patched_state, sniff_model_type

from .gradients import BlockLayerGradient


class DareUnetMerger:
    """
    A class to merge two diffusion U-Net models using calculated deltas, sparsification,
    and a weighted consensus method. This is the DARE-TIES method.
    
    https://arxiv.org/pdf/2311.03099.pdf
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
                "drop_rate": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ties": (["sum", "count", "off"], {"default": "sum"}),
                "rescale": (["off", "on"], {"default": "off"}),
                "seed": ("INT", {"default": 1, "min":0, "max": 99999999999}),
                "method": (["comfy",] + METHODS, {"default": "comfy"}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "model_mask": (MODEL_MASK,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = UNET_CATEGORY

    def merge(self, model_a: ModelPatcher, **kwargs,) -> Tuple[ModelPatcher]:
        """
        Merges two ModelPatcher instances based on the weighted consensus of their parameters and sparsity.

        Args:
            model_a (ModelPatcher): The base model to be merged.
            model_b (ModelPatcher): The model to merge into the base model.
            method (str): The method to use for merging, either "lerp", "slerp", or "gradient".
            seed (int): The random seed to use for the merge.
            clear_cache (bool): Whether to clear the CUDA cache after each chunk. Default is False.
            iterations (int): The number of iterations to perform the merge.  Default is 1.
            model_mask (Optional[ModelMask]): The model mask to use for protection of our model_a. Default is None.
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the merged ModelPatcher instance.
        """

        gradient = BlockLayerGradient().gradient(model=model_a, **kwargs)[0]
        return DareUnetMergerGradient().merge(model_a=model_a, gradient=gradient, **kwargs)
        

class DareUnetMergerGradient:
    """
    A class to merge two diffusion U-Net models using calculated deltas, sparsification,
    and a weighted consensus method. This is the DARE-TIES method.
    
    https://arxiv.org/pdf/2311.03099.pdf
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
                "gradient": (LAYER_GRADIENT,),
                "drop_rate": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ties": (["sum", "count", "off"], {"default": "sum"}),
                "rescale": (["off", "on"], {"default": "off"}),
                "seed": ("INT", {"default": 1, "min":0, "max": 99999999999}),
                "method": (["comfy",] + METHODS, {"default": "comfy"}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "model_mask": (MODEL_MASK,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = UNET_CATEGORY

    def merge(self, model_a: ModelPatcher, model_b: ModelPatcher, gradient : Dict[str, float],
              method : str, seed : Optional[int] = None, clear_cache : bool = True,
              model_mask: Optional[ModelMask] = None, iterations : int = 1,
              **kwargs,) -> Tuple[ModelPatcher]:
        """
        Merges two ModelPatcher instances based on the weighted consensus of their parameters and sparsity.

        Args:
            model_a (ModelPatcher): The base model to be merged.
            model_b (ModelPatcher): The model to merge into the base model.
            method (str): The method to use for merging, either "lerp", "slerp", or "gradient".
            seed (int): The random seed to use for the merge.
            clear_cache (bool): Whether to clear the CUDA cache after each chunk. Default is False.
            iterations (int): The number of iterations to perform the merge.  Default is 1.
            model_mask (Optional[ModelMask]): The model mask to use for protection of our model_a. Default is None.
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the merged ModelPatcher instance.
        """

        device = get_device()

        if seed is not None:
            torch.manual_seed(seed)
        
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        with cuda_memory_profiler():
            m = model_a.clone()  # Clone model_a to keep its structure
            model_a_sd = get_patched_state(m)
            model_b_sd = get_patched_state(model_b)

            # Merge each parameter from model_b into model_a
            for k in model_a_sd.keys():
                if k not in model_b_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                ratio = gradient.get(k, None)
                if ratio is None:
                    continue
                elif ratio == 1.0:
                    continue

                # Apply sparsification by the delta for this layer
                mask : torch.Tensor = model_mask.get_layer_mask(k) if model_mask is not None else None
                a : torch.Tensor = model_a_sd[k]
                b : torch.Tensor = model_b_sd[k]

                merged_a = a.clone()

                for i in range(iterations):
                    if seed is not None:
                        torch.manual_seed(seed + i)

                    #print("merging key:", k, "with ratio:", ratio, "and method:", method, "and mask:", mask.shape, mask.dtype, mask.sum().item())

                    sparsified_delta = dare_ties_sparsification(merged_a, b, device=device, **kwargs)
                    # If we have a mask, apply it to the delta, replacing true values with our delta
                    if mask is not None:
                        #print(f"Merging {k} with mask Included: {torch.count_nonzero(mask)}, Excluded: {torch.count_nonzero(~mask)}")
                        sparsified_delta = torch.where(mask.to(device), sparsified_delta.to(device), merged_a.to(device))

                    if method == "comfy":
                        merged_a = sparsified_delta

                        strength_patch = 1.0 - ratio
                        strength_model = ratio
                    else:
                        merged_a = merge_tensors(method, merged_a.to(device), sparsified_delta.to(device), 1 - ratio)

                        strength_model = 0
                        strength_patch = 1.0

                del sparsified_delta
                
                # Apply the sparsified delta as a patch
                m.add_patches({k: (merged_a.to('cpu'),)}, strength_patch, strength_model)

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (m,)
    
    @classmethod
    def get_layer_profile(cls, sd : Dict[str, torch.Tensor], exterior : float, interior : float, core : float, **kwargs) -> Dict[str, float]:
        # SD1.5 has 12 input and output blocks, and 1 middle block
        # exterior - input_blocks.{0, 1, 2, 3} and output_blocks.{9, 10, 11, 12}
        # interior - input_blocks.{4, 5, 6, 7, 8} and output_blocks.{4, 5, 6, 7, 8}
        # core - input_blocks.{9, 10, 11, 12} middle_block and output_blocks.{0, 1, 2, 3}
        # SDXL has 9 input and output blocks, and 1 middle block
        # exterior - input_blocks.{0, 1, 2} and output_blocks.{6, 7, 8}
        # interior - input_blocks.{3, 4, 5} and output_blocks.{3, 4, 5}
        # core - input_blocks.{6, 7, 8} middle_block and output_blocks.{0, 1, 2}
        
        model_type = sniff_model_type(sd)
        if model_type == "sd15":
            spec = {
                'exterior': [[0,1,2,3], None, [9,10,11,12]],
                'interior': [[4,5,6,7,8], False, [4,5,6,7,8]],
                'core': [[9,10,11,12], True, [0,1,2,3]],
            }
        elif model_type == "sdxl":
            spec = {
                'exterior': [[0,1,2], False, [6,7,8]],
                'interior': [[3,4,5], False, [3,4,5]],
                'core': [[6,7,8], True, [0,1,2]],
            }
        else:
            raise RuntimeError(f"Unknown model type {model_type}")

        result = {}
        for k in spec.keys():
            ratio = {
                'exterior': exterior,
                'interior': interior,
                'core': core,
            }.get(k)
            for n in spec[k][0]:
                result[f"diffusion_model.input_blocks.{n}"] = ratio
            if spec[k][1] is not None:
                result[f"diffusion_model.middle_block."] = ratio
            for n in spec[k][2]:
                result[f"diffusion_model.output_blocks.{n}"] = ratio

            
        return result

    @classmethod
    def calculate_layer_ratio(cls, key : str, profile : Dict[str, float],
                              process_norm : bool, norm : float,
                              process_attn : bool, attn : float,
                              process_ff_net : bool, ff_net : float,
                              **kwargs) -> Optional[float]:
        ratio = None

        for k in profile.keys():
            if key.startswith(k):
                ratio = profile[k]
                break

        if ratio is None:
            return None

        # Get our ratio for this layer
        if "norm" in key and process_norm:
            ratio *= norm
        elif "attn" in key and process_attn:
            ratio *= attn
        elif "ff.net" in key and process_ff_net:
            ratio *= ff_net
        else:
            return None

        return ratio
