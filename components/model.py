# components/model.py
from comfy.model_patcher import ModelPatcher
import torch
from typing import Dict, Tuple, Optional

from ..ddare.const import UTIL_CATEGORY, MODEL_MASK
from ..ddare.mask import ModelMask
from ..ddare.merge import merge_tensors, METHODS
from ..ddare.model import collect_layers, layers_for_mask
from ..ddare.util import cuda_memory_profiler, get_device, get_patched_state


class ModelNoiseInjector:
    """
    Inject gaussian noise into layers of a model.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the noise injection process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "model": ('MODEL',),
                "operation": (["random", "gaussian"], {'default': "gaussian"}),
                "ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mean": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "std": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min":0, "max": 99999999999}),
                # It would be cool if we could autopupulate layer names here from a dropdown
                "layers": ("STRING", {"multiline": True}),
                "method": (["comfy",] + METHODS, {"default": "comfy"}),
            },
            "optional": {
                "model_mask": (MODEL_MASK,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "edit"
    CATEGORY = UTIL_CATEGORY
    
    def edit(self, model : ModelPatcher, operation : str, mean : float, std : float, ratio : float, seed : int, layers : str, method : str, model_mask : Optional[ModelMask] = None, **kwargs) -> Tuple[ModelPatcher]:
        """
        Injects noise into the model.

        Args:
            model (ModelPatcher): The model to inject noise into.
            operation (str): The type of noise to inject.  Either "random" or "gaussian".
            mean (float): The first argument for the noise injection. Only used for "gaussian" noise.
            std (float): The second argument for the noise injection. Only used for "gaussian" noise.
            ratio (float): The strength of the noise to inject.
            seed (int): The seed for the noise injection.
            layers (str): The layers to inject noise into.
            model_mask (ModelMask): A ModelMask instance to use for masking the model. Default is None.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the modified ModelPatcher instance.
        """

        device = get_device()
        
        m = model.clone()  # Clone model_a to keep its structure

        with cuda_memory_profiler():
            model_sd = get_patched_state(m)

            keys = list(model_sd.keys())
            collected_targets = collect_layers(layers, keys)
            if len(collected_targets) == 0:
                raise ValueError("No layers specified")

            for i, target in enumerate(collected_targets):
                for j, k in enumerate(layers_for_mask(target, keys)):
                    if k not in model_sd:
                        print("could not patch. key doesn't exist in model:", k)
                        continue

                    torch.manual_seed(seed + i * 1000 + j)

                    # Get our tensor
                    mask : torch.Tensor = model_mask.get_layer_mask(k) if model_mask is not None else None
                    a : torch.Tensor = model_sd[k]

                    if operation == "random":
                        # Create a random mask of the same shape as the given layer.
                        random = torch.rand(a.shape) - 0.5
                    else:
                        # Create a gaussian noise mask of the same shape as the given layer.
                        random = torch.normal(mean, std, size=a.shape) - mean
                        
                    if mask is None:
                        result_tensor = a + random
                    else:
                        result_tensor = torch.where(mask.to(device), a.to(device) + random.to(device), a.to(device))
                    del random

                    # Merge our tensors
                    if method == "comfy":
                        strength_patch = 1.0 - ratio
                        strength_model = ratio
                    else:
                        result_tensor = merge_tensors(method, a.to(device), result_tensor, 1 - ratio)

                        strength_model = 0
                        strength_patch = 1.0

                    m.add_patches({k: (result_tensor.to('cpu'),)}, strength_patch, strength_model)

        return (m,)