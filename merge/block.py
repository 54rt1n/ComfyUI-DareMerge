# merge/block.py
import torch
from typing import Dict, Tuple, Optional

from comfy.model_patcher import ModelPatcher

from .mergeutil import merge_tensors


class BlockModelMergerAdv:
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
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "ddare/block"

    def merge(self, model_a: ModelPatcher, model_b: ModelPatcher, method : str, input : float, middle : float, out : float, time : float, **kwargs) -> Tuple[ModelPatcher]:
        """
        Merges two ModelPatcher instances based on the weighted consensus of their parameters and sparsity.

        Args:
            model_a (ModelPatcher): The base model to be merged.
            model_b (ModelPatcher): The model to merge into the base model.
            input (float): The ratio (lambda) of the input layer to keep from model_a.
            middle (float): The ratio (lambda) of the middle layers to keep from model_a.
            out (float): The ratio (lambda) of the output layer to keep from model_a.
            time (float): The ratio (lambda) of the time layer to keep from model_a.
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

            merged_layer = merge_tensors(method, a.to(device), b.to(device), 1 - ratio)

            nv = (merged_layer.to('cpu'),)

            del a, b
            
            # We have already merged the models
            m.add_patches({k: nv}, 1, 0)

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

