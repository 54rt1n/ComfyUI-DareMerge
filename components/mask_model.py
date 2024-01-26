# components/mask_model.py
from collections import defaultdict
from comfy.model_patcher import ModelPatcher
import re
import torch
from typing import Dict, Tuple, List, Generator, Optional

from ..ddare.const import MASK_CATEGORY
from ..ddare.mask import ModelMask
from ..ddare.tensor import get_threshold_mask, bernoulli_noise, gaussian_noise
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
                "mask_a": ("MODEL_MASK",),
                "mask_b": ("MODEL_MASK",),
                "operation": (["union", "intersect", "difference", "xor"], {"default": "union"}),
            }
        }
        
    RETURN_TYPES = ("MODEL_MASK",)
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


class MaskReporting:
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
                "mask": ("MODEL_MASK",),
                "report": (["size", "details"], {"default": "size"}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    FUNCTION = "mask_report"
    CATEGORY = MASK_CATEGORY
    
    def mask_report(self, mask: ModelMask, report: str = "size", **kwargs) -> Tuple[str]:
        """
        Generate a report on the mask.

        Args:
            mask (ModelMask): The mask.
            report (str): The report to generate. 

        Returns:
            Tuple[str]: A tuple containing the report.
        """
        if report == "size":
            return (self.size_report(mask), )
        if report == "details":
            return (self.list_layers(mask), )
        else:
            raise ValueError("Unknown report: {}".format(report))

    def size_report(self, mask: ModelMask) -> Tuple[str]:
        """
        Generate a report on the size of the mask.

        Args:
            mask (ModelMask): The mask.

        Returns:
            Tuple[str]: A tuple containing the report.
        """
        sd = mask.state_dict
        data = defaultdict(dict)
        for k in sd.keys():
            parts = k.split(".", 2)
            if len(parts) == 2:
                print("skipping", k)
            else:
                model, block, rest = parts
                # our report is a tuple containing the number of elements, and the number of elements that are true
                data[block][rest] = (sd[k].numel(), sd[k].sum().item())

        report = ""
        for block in data.keys():
            total = 0
            total_true = 0
            for rest in data[block].keys():
                total += data[block][rest][0]
                total_true += data[block][rest][1]
            report += f"{block}: {total_true} / {total} ({total_true / total * 100:.2f}%)\n"
        
        return report

    def list_layers(self, mask : ModelMask) -> Tuple[str]:
        """

        Args:
            mask (ModelMask): _description_

        Returns:
            Tuple[str]: _description_
        """

        result = ""
        for k in sorted(mask.state_dict.keys(), key=lambda x: self.sort_key_for_zero_padding(x)):
            size = mask.state_dict[k].numel()
            true = mask.state_dict[k].sum().item()
            result += f"{k}: {true} / {size} ({true / size * 100:.2f}%)\n"
        return (result,)

    def sort_key_for_zero_padding(self, s: str, width : int = 2) -> str:
        """
        Custom sort key function that uses zero-padding for sorting.

        Args:
            s (str): The string to be sorted.
            width (int): The fixed width for zero-padding numbers.

        Returns:
            str: A zero-padded string for sorting purposes.
        """
        return re.sub(r'(\d+)', lambda x: x.group(1).zfill(width), s)


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
                "mask": ("MODEL_MASK",),
                "operation": (["random", "gaussian", "true", "false"], {'default': "random"}),
                "arg_one": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "arg_two": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min":0, "max": 99999999999}),
                # It would be cool if we could autopupulate layer names here from a dropdown
                "layers": ("STRING", {"multiline": True}),
            },
        }
        
    RETURN_TYPES = ("MODEL_MASK",)
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
        
        collected_targets = self.collect_layers(layers, mask)
        if len(collected_targets) == 0:
            raise ValueError("No layers specified")

        mask = mask.clone()
        if seed is not None:
            torch.manual_seed(seed)
        for target in collected_targets:
            for layer in self.layers_for_mask(target, mask):
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

    def layer_in_mask(self, layer: str, mask: ModelMask) -> bool:
        """
        Checks if a layer is in the mask.

        Args:
            layer (str): The layer name.
            mask (ModelMask): The mask.

        Returns:
            bool: True if the layer is in the mask.
        """
        for k in self.layers_for_mask(layer, mask):
            return True
        return False

    def layers_for_mask(self, layer: str, mask: ModelMask) -> Generator[str, None, None]:
        """
        Gets the layers for a mask.

        Args:
            layer (str): The layer name.
            mask (ModelMask): The mask.

        Returns:
            Generator[str, None, None]: A generator containing the layers.
        """
        keys = list(mask.state_dict.keys())
        # If layer doesn't end with a dot, add one
        if not layer.endswith("."):
            layer += "."
        
        # Match our wildcard
        if re.search(r"\*", layer):
            # We need to escape the layer name, and then replace the wildcard with a regex
            wclayer = re.escape(layer)
            wclayer = re.sub(r"\\\*", r"(.*)", wclayer)
            wclayer = re.compile(wclayer)
        else:
            wclayer = None
        
        for k in keys:
            #prefix = len("diffusion_model.")
            prefix = 0
            key = k[prefix:]
            if key.startswith(layer) or key.endswith(layer) or key == layer:
                yield k
            elif wclayer is not None:
                match = wclayer.match(key)
                if match:
                    yield k

    def collect_layers(self, layers: str, mask: ModelMask) -> List[str]:
        """
        Collects the layer names from the input string.

        Args:
            layers (str): The layer names.

        Returns:
            Tuple[str]: A tuple containing the layer names.
        """
        # We should split by newline and comma, and remove whitespace and empty strings
        clean = re.sub(r"\s+", "", layers)
        layers = re.split(r"[\n,]", clean)
        # if we have any braces, we need to collect the numbers inside and expand them
        # we do this by matching for braces, and then pulling out the comma separated values inside with regex
        # TODO recurse this to handle multiple braces in one key
        
        bracket = re.compile(r"\{(.*?)\}")
        results = []
        for layer in layers:
            match = bracket.match(layer)
            if match:
                matchval = match.group(1)
                branches = re.sub(r"\s+", "", matchval).split(",")
                for branch in branches:
                    new_branch = re.sub(layer, matchval, branch)
                    if self.layer_in_mask(new_branch, mask):
                        results.append(new_branch)
                    else:
                        print("Branch not found, skipping", new_branch)
            else:
                if self.layer_in_mask(layer, mask):
                    results.append(layer)
                else:
                    print("Layer not found, skipping", layer)
        return results


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

    RETURN_TYPES = ("MODEL_MASK",)
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
