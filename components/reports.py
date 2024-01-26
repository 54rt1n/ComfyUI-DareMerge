# components/reports.py

from comfy.model_patcher import ModelPatcher
from collections import defaultdict
import re
from typing import Dict, Tuple

from ..ddare.const import REPORT_CATEGORY
from ..ddare.mask import ModelMask
from ..ddare.reporting import plot_model_layer, PLOT_SCALING
from ..ddare.util import get_patched_state


def sort_key_for_zero_padding(s: str, width : int = 2) -> str:
    """
    Custom sort key function that uses zero-padding for sorting.

    Args:
        s (str): The string to be sorted.
        width (int): The fixed width for zero-padding numbers.

    Returns:
        str: A zero-padded string for sorting purposes.
    """
    return re.sub(r'(\d+)', lambda x: x.group(1).zfill(width), s)


class MaskReporting:
    """
    Generate some text reports on the mask.
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
    CATEGORY = REPORT_CATEGORY
    
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
        for k in sorted(mask.state_dict.keys(), key=lambda x: sort_key_for_zero_padding(x)):
            size = mask.state_dict[k].numel()
            true = mask.state_dict[k].sum().item()
            result += f"{k}: {true} / {size} ({true / size * 100:.2f}%)\n"
        return (result,)


class ModelReporting:
    """
    Dump our plots to a file.
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
                "layer": ('STRING', {"default": ""}),
                "scaling": (PLOT_SCALING, {"default": "mean"}),
            }
        }
        
    RETURN_TYPES = ("STRING","IMAGE",)
    FUNCTION = "model_report"
    CATEGORY = REPORT_CATEGORY
    
    def model_report(self, model: ModelPatcher, layer: str, scaling : str, **kwargs) -> Tuple[str]:
        """
        Generate a report on the model.

        Args:
            model (ModelPatcher): The model.
            report (str): The report to generate. 

        Returns:
            Tuple[str]: A tuple containing the report.
        """

        sd = get_patched_state(model)
        
        if layer not in sd:
            raise ValueError("Layer {} not found in model".format(layer))
        
        l = sd[layer]

        image = plot_model_layer(l, layer, scaling=scaling, show_legend=True)

        return (None, [image])
