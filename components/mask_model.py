# components/mask_model.py
from collections import defaultdict
from comfy.model_patcher import ModelPatcher
import torch
from typing import Dict, Tuple

from ..ddare.const import MASK_CATEGORY
from ..ddare.mask import ModelMask
from ..ddare.tensor import get_threshold_mask
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
                "report": (["size"], {"default": "size"}),
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