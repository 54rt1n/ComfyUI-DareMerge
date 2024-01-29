# components/mask_model.py
from comfy.model_patcher import ModelPatcher
import torch
from typing import Dict, Tuple, Optional

from ..ddare.const import MASK_CATEGORY, MODEL_MASK
from ..ddare.mask import ModelMask
from ..ddare.model import layers_for_mask, collect_layers
from ..ddare.tensor import get_threshold_mask, bernoulli_noise, gaussian_noise, divide_tensor_into_sets
from ..ddare.util import cuda_memory_profiler, get_device, get_patched_state


class MagnitudeMasker:
    """
    A managed state dict to allow for masking of model layers.  This is used for protecting
    layers from being overwritten by the merge process.
    """

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
                "select": (["above", "below"], {"default": "below"}),
            }
        }

    RETURN_TYPES = (MODEL_MASK,)
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
            model_a_sd = get_patched_state(model_a)
            model_b_sd = get_patched_state(model_b)

            mm = ModelMask({})

            # Merge each parameter from model_b into model_a
            for k in model_a_sd.keys():
                if k not in model_b_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                a : torch.Tensor = model_a_sd[k]
                b : torch.Tensor = model_b_sd[k]

                layer_mask = get_threshold_mask(a, b, device=device, **kwargs)
                mm.add_layer_mask(k, layer_mask)

        return (mm,)


class MaskOperations:
    """
    Take two masks and perform a set operation.  union, intersect, difference, xor
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the masking process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "mask_a": (MODEL_MASK,),
                "mask_b": (MODEL_MASK,),
                "operation": (["union", "intersect", "difference", "xor"], {"default": "union"}),
            }
        }
        
    RETURN_TYPES = (MODEL_MASK,)
    FUNCTION = "mask_ops"
    CATEGORY = MASK_CATEGORY
    
    def mask_ops(self, mask_a: ModelMask, mask_b: ModelMask, operation: str = "union", **kwargs) -> Tuple[ModelMask]:
        """
        Take two masks and perform a set operation.  union, intersect, difference, xor

        Args:
            mask_a (ModelMask): The first mask.
            mask_b (ModelMask): The second mask.
            operation (str): The operation to perform.

        Returns:
            Tuple[ModelMask]: A tuple containing the mask.
        """
        if operation == "union":
            return (ModelMask.union(mask_a, mask_b),)
        elif operation == "intersect":
            return (ModelMask.intersect(mask_a, mask_b),)
        elif operation == "difference":
            return (ModelMask.set_difference(mask_a, mask_b),)
        elif operation == "xor":
            return (ModelMask.symmetric_distance(mask_a, mask_b),)
        else:
            raise ValueError("Unknown operation: {}".format(operation))


class MaskEdit:
    """
    Takes a mask and edits it.  This is where all the hacks happen.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the masking process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "mask": (MODEL_MASK,),
                "operation": (["random", "gaussian", "true", "false"], {'default': "random"}),
                "arg_one": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "arg_two": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min":0, "max": 99999999999}),
                # It would be cool if we could autopupulate layer names here from a dropdown
                "layers": ("STRING", {"multiline": True}),
            },
        }
        
    RETURN_TYPES = (MODEL_MASK,)
    FUNCTION = "mask_command"
    CATEGORY = MASK_CATEGORY
    
    def mask_command(self, mask: ModelMask, operation: str = "random", arg_one: float = 0.0, arg_two: float = 0.0, seed: Optional[int] = None, layers: str = "", **kwargs) -> Tuple[ModelMask]:
        """
        Run a command on the mask.
        
        Args:
            mask (ModelMask): The mask.
            operation (str): The operation to perform.
            arg_one (float): The first argument.
            arg_two (float): The second argument.
            seed (int): The random seed.
            layers (str): The layer names.  This can be a comma separated list, with wildcards, or using {0, 1} to target specific layers.
            
        Returns:
            Tuple[ModelMask]: A tuple containing the mask.
        """
        
        keys = list(mask.state_dict.keys())
        collected_targets = collect_layers(layers, keys)
        if len(collected_targets) == 0:
            raise ValueError("No layers specified")

        mask = mask.clone()
        if seed is not None:
            torch.manual_seed(seed)
        for target in collected_targets:
            for layer in layers_for_mask(target, keys):
                print("Editing", layer, operation)
                if operation == "random":
                    mask.noise_layer(layer, "bernoulli", v0=arg_one, v1=arg_two)
                elif operation == "gaussian":
                    mask.noise_layer(layer, "gaussian", v0=arg_one, v1=arg_two)
                elif operation == "true":
                    mask.boolean_layer(layer, True)
                elif operation == "false":
                    mask.boolean_layer(layer, False)
                else:
                    raise ValueError("Unknown operation: {}".format(operation))
            
        return (mask,)


class SimpleMasker:
    """
    A managed state dict to allow for masking of model layers.  This is used for protecting
    layers from being overwritten by the merge process.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the masking process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "model": ("MODEL",),
                "operation": (["random", "gaussian", "true", "false"], {'default': "true"}),
                "arg_one": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "arg_two": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min":0, "max": 99999999999}),
            }
        }

    RETURN_TYPES = (MODEL_MASK,)
    FUNCTION = "mask"
    CATEGORY = MASK_CATEGORY

    def mask(self, model: ModelPatcher, operation: str = "true", arg_one: float = 0.0, arg_two: float = 0.0, seed: int = 1, **kwargs) -> Tuple[ModelMask]:
        """
        Takes a model and creates a mask for it using one of the following methods:
        - random: Creates a bournoulli mask with the given threshold.
        - gaussian: Creates a gaussian mask with the given mean and standard deviation.
        - true: Creates a mask of all true values.
        - false: Creates a mask of all false values.

        Args:
            model (ModelPatcher): The model to create a mask for.
            operation (str): The type of mask to create.
            arg_one (float): The first argument.  For random, this is the threshold.  For gaussian, this is the mean.
            arg_two (float): The second argument.  For gaussian, this is the standard deviation.
            seed (int): The random seed.  Only used for random and gaussian masks.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the mask.
        """
        model_sd = model.model_state_dict()

        mm = ModelMask({})

        # Merge each parameter from model_b into model_a
        for i, k in enumerate(model_sd.keys()):
            if seed is not None:
                torch.manual_seed(seed + i)
            if operation == "random":
                mask = bernoulli_noise(model_sd[k], arg_one)
            elif operation == "gaussian":
                mask = gaussian_noise(model_sd[k], arg_one, arg_two)
            elif operation == "true":
                mask = torch.ones(model_sd[k].shape)
            elif operation == "false":
                mask = torch.zeros(model_sd[k].shape)
            else:
                raise ValueError("Unknown operation: {}".format(operation))

            mm.add_layer_mask(k, mask)

        return (mm,)


class QuadMasker:
    """
    A managed state dict to allow for masking of model layers.  This is used for protecting
    layers from being overwritten by the merge process.  This node creates four masks, each
    from a different slices of the model parameters.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the masking process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 1, "min":0, "max": 99999999999}),
            }
        }

    RETURN_TYPES = (MODEL_MASK, MODEL_MASK, MODEL_MASK, MODEL_MASK,)
    FUNCTION = "mask"
    CATEGORY = MASK_CATEGORY

    def mask(self, model: ModelPatcher, seed: int = 1, **kwargs) -> Tuple[ModelMask]:
        """
        Takes a model and creates several masks for it.

        Args:
            model (ModelPatcher): The model to create a mask for.
            seed (int): The random seed.  Only used for random and gaussian masks.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the mask.
        """

        model_sd = model.model_state_dict()
        mask_count = 4

        mms = [ModelMask({}) for _ in range(mask_count)]

        # Merge each parameter from model_b into model_a
        for i, k in enumerate(model_sd.keys()):
            tensor = model_sd[k]
            sets = divide_tensor_into_sets(tensor, mask_count)

            for j in range(mask_count):
                if seed is not None:
                    torch.manual_seed(seed + i * mask_count + j)
                mask = sets == j
                mms[j].add_layer_mask(k, mask)

        return (*mms,)

