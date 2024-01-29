# components/reports.py

from comfy.model_patcher import ModelPatcher
from collections import defaultdict
import folder_paths
import json
from typing import Dict, Tuple

from ..ddare.const import REPORT_CATEGORY, LAYER_GRADIENT, MODEL_MASK
from ..ddare.lora import DoctorLora
from ..ddare.mask import ModelMask
from ..ddare.reporting import plot_model_layer, PLOT_SCALING
from ..ddare.util import get_patched_state, sort_key_for_zero_padding, dumb_json


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
                "mask": (MODEL_MASK,),
                "report": (["size", "details"], {"default": "size"}),
            }
        }
        
    RETURN_TYPES = ("STRING","IMAGE",)
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


class LoRAReporting:
    """
    Generate some reports on a LoRA
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
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "report": (["tags", "size", "details"], {"default": "tags"}),
                "limit": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING","IMAGE",)
    FUNCTION = "lora_report"
    CATEGORY = REPORT_CATEGORY

    SIZE_KEYS = ["dm_filename", "dm_filesize", "ss_clip_skip", "ss_epoch", "ss_mixed_precision", "ss_network_args", "ss_noise_offset", "ss_num_batches_per_epoch", "ss_num_epochs", "ss_num_train_images", "ss_output_name", "ss_resolution", "ss_sd_model_hash", "ss_sd_model_name", "ss_steps", "ss_total_batch_size", "ss_training_comment", "sshs_model_hash"]
    
    def lora_report(self, lora_name : str, report: str = "size", **kwargs) -> Tuple[str]:
        """
        Generate a report on the LoRA.

        Args:
            lora (LoRA): The LoRA.
            report (str): The report to generate. 

        Returns:
            Tuple[str]: A tuple containing the report.
        """
        lora_path = folder_paths.get_full_path("loras", lora_name)
        dl = DoctorLora.load(lora_path)
        if dl is None:
            raise ValueError("Could not load LoRA file: {}".format(lora_name))
        if report == "size":
            report = ""
            report += f"Key Count: {dl.keycount}\n"
            report += f"Parameters: {dl.parameters}\n"
            for k in sorted(dl.metadata.keys(), key=sort_key_for_zero_padding):
                if k == "dm_signature":
                    for k2, v2 in dl.metadata[k].items():
                        report += f"{k2}: {v2}\n"
                elif k in self.SIZE_KEYS:
                    report += f"{k}: {dl.metadata[k]}\n"
                        
            return (report, )
        if report == "details":
            return (json.dumps(dl.metadata, default=dumb_json), )
        if report == "tags":
            def tags_report(limit : int = 10, **myargs):
                tags = dl.tags
                if len(tags) == 0:
                    return "No tags"
                return ", ".join(tags[:limit])
            return (tags_report(**kwargs), )
        else:
            raise ValueError("Unknown report: {}".format(report))


class LayerGradientReporting:
    """
    Generate some reports on a gradient
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
                "gradient": (LAYER_GRADIENT,),
                "report": (["size", "details"], {"default": "size"}),
            }
        }

    RETURN_TYPES = ("STRING","IMAGE",)
    FUNCTION = "gradient_report"
    CATEGORY = REPORT_CATEGORY

    def gradient_report(self, gradient : Dict[str, float], report: str = "size", **kwargs) -> Tuple[str]:
        """
        Generate a report on the gradient.

        Args:
            gradient (Dict[str, float]): The gradient.
            report (str): The report to generate. 

        Returns:
            Tuple[str]: A tuple containing the report.
        """
        if report == "size":
            return (self.size_report(gradient), )
        if report == "details":
            return (self.list_layers(gradient), )
        else:
            raise ValueError("Unknown report: {}".format(report))

    def size_report(self, gradient : Dict[str, float]) -> Tuple[str]:
        """
        Generate a report on the size of the gradient.

        Args:
            gradient (Dict[str, float]): The gradient.

        Returns:
            Tuple[str]: A tuple containing the report.
        """
        report = ""
        for k in sorted(gradient.keys(), key=lambda x: sort_key_for_zero_padding(x)):
            report += f"{k}: {gradient[k]:.2f}\n"
        
        return report

    def list_layers(self, gradient : Dict[str, float]) -> Tuple[str]:
        """

        Args:
            gradient (Dict[str, float]): _description_

        Returns:
            Tuple[str]: _description_
        """

        result = ""
        for k in sorted(gradient.keys(), key=lambda x: sort_key_for_zero_padding(x)):
            result += f"{k}: {gradient[k]:.2f}\n"
        return (result,)