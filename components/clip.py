# components/clip.py
from comfy.sd import CLIP
import torch
from typing import Optional

from ..ddare.const import CLIP_CATEGORY
from ..ddare.merge import merge_tensors
from ..ddare.tensor import dare_ties_sparsification
from ..ddare.util import cuda_memory_profiler, get_device


class DareClipMerger:
    """
    A class to merge two CLIP models using calculated deltas, sparsification,
    and a weighted consensus method. This is the DARE-TIES method.
    
    https://arxiv.org/pdf/2311.03099.pdf
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_a": ("CLIP",),
                              "clip_b": ("CLIP",),
                              "ties": (["sum", "count", "off"], {"default": "sum"}),
                              "rescale": (["off", "on"], {"default": "off"}),
                              "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "drop_rate": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "seed": ("INT", {"default": 42}),
                              "method": (["comfy", "lerp", "slerp", "gradient"], ),
                              "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                            }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "merge"
    CATEGORY = CLIP_CATEGORY

    def merge(self, clip_a : CLIP, clip_b : CLIP, ratio : float, method : str, seed : Optional[int] = None, iterations : int = 1, clear_cache : bool = True, **kwargs):
        """
        Merge two CLIP models using the DARE-TIES method.

        Args:
            clip_a (CLIP): The base CLIP model
            clip_b (CLIP): The CLIP model to merge into the base
            ratio (float): The ratio of the models to use.  1 is 100% model_a, 0 is 100% model_b.
            method (str): The merge method to use
            seed (Optional[int]): The seed to use for the random number generator
            iterations (int): The number of iterations to use
            clear_cache (bool): Whether to clear the GPU cache after merging
            **kwargs: Unused

        Returns:
        """
        device = get_device()
        
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with cuda_memory_profiler():
            clip_a = clip_a.clone()
            clip_a.patcher.patch_model()
            clip_a_sd = clip_a.get_sd()  # State dict of model_a
            clip_a.patcher.unpatch_model()
            clip_b = clip_b.clone()
            clip_b.patcher.patch_model()
            clip_b_sd = clip_b.get_sd()
            clip_b.patcher.unpatch_model()

            for k in clip_a_sd.keys():
                if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                    continue

                if k not in clip_a_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                a : torch.Tensor = clip_a_sd[k]
                b : torch.Tensor = clip_b_sd[k]

                merged_a = a.clone()

                for i in range(iterations):
                    if seed is not None:
                        torch.manual_seed(seed + i)
                    sparsified_delta = dare_ties_sparsification(merged_a, b, device=device, **kwargs)

                    if method == "comfy":
                        merged_a = sparsified_delta

                        strength_patch = 1.0 - ratio
                        strength_model = ratio
                    else:
                        merged_a = merge_tensors(method, merged_a.to(device), sparsified_delta.to(device), 1 - ratio)

                        strength_model = 0
                        strength_patch = 1.0

                    del sparsified_delta

                #del a, b
                
                # Apply the sparsified delta as a patch
                nv = (merged_a.to('cpu'),)

                clip_a.add_patches({k: nv}, strength_patch, strength_model)

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (clip_a,)
