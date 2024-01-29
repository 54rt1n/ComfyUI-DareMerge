# components/gradients.py

from comfy.model_patcher import ModelPatcher
import torch
from typing import Dict, Tuple, Optional

from ..ddare.const import GRADIENT_CATEGORY, LAYER_GRADIENT
from ..ddare.model import collect_layers, layers_for_mask
from ..ddare.util import sniff_model_type

class ShellLayerGradient:
    """
    A class to hold a layer gradient for a model.
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
                "model": ("MODEL",),
                "exterior": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interior": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "core": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (LAYER_GRADIENT,)
    FUNCTION = "gradient"
    CATEGORY = GRADIENT_CATEGORY
    
    def gradient(self, model : ModelPatcher, exterior : float, interior : float, core : float, **kwargs) -> Tuple[Dict[str, float]]:
        """
        Returns a gradient dictionary for the specified model type.

        Args:
            model (ModelPatcher): The model to generate the gradient for.
            exterior (float): The gradient for the exterior layers.
            interior (float): The gradient for the interior layers.
            core (float): The gradient for the core layers.

        Returns:
            Tuple[Dict[str, float]]: A dictionary of layer gradients.
        """
        sd : Dict[str, torch.Tensor] = model.model_state_dict()
        profile = self.get_layer_profile(sd, exterior, interior, core, **kwargs)
        gradient : Dict[str, float] = {}
        for k in sd.keys():
            ratio = None
            for pk in profile.keys():
                if k.startswith(pk):
                    ratio = profile[pk]
                    break

            if ratio is None:
                continue

            gradient[k] = ratio

        return (gradient, )

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


class AttentionLayerGradient:
    """
    A class to hold a layer gradient for a model targeting the attention layers.
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
                "model": ("MODEL",),
                "process_norm": ("BOOLEAN", {"default": True}),
                "process_attn": ("BOOLEAN", {"default": True}),
                "process_ff_net":("BOOLEAN", {"default": True}),
                "norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ff_net": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (LAYER_GRADIENT,)
    FUNCTION = "gradient"
    CATEGORY = GRADIENT_CATEGORY
    
    def gradient(self, model : ModelPatcher, 
                 process_norm : bool, norm : float,
                 process_attn : bool, attn : float,
                 process_ff_net : bool, ff_net : float,
                 **kwargs) -> Tuple[Dict[str, float]]:
        """
        Returns a gradient dictionary for the specified model type.

        Args:
            model (ModelPatcher): The model to generate the gradient for.
            process_norm (bool): Whether to process the norm layers.
            norm (float): The gradient for the norm layers.
            process_attn (bool): Whether to process the attention layers.
            attn (float): The gradient for the attention layers.
            process_ff_net (bool): Whether to process the feed forward layers.
            ff_net (float): The gradient for the feed forward layers.

        Returns:
            Tuple[Dict[str, float]]: A dictionary of layer gradients.
        """
        sd : Dict[str, torch.Tensor] = model.model_state_dict()
        gradient : Dict[str, float] = {}
        for k in sd.keys():
            ratio = None
            if "norm" in k and process_norm:
                ratio = norm
            elif "attn" in k and process_attn:
                ratio = attn
            elif "ff.net" in k and process_ff_net:
                ratio = ff_net

            if ratio is None:
                continue
            gradient[k] = ratio

        return (gradient, )


class LayerGradientOperations:
    """
    Run operations on two graidents.  Add, subtract, mulitply, divide, etc.
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
                "gradient_a": (LAYER_GRADIENT,),
                "gradient_b": (LAYER_GRADIENT,),
                "operation": (["mean", "min", "max", "add", "subtract", "multiply", "divide"],{ "default": "mean" }),
                "join": (["inner", "outer"], {"default": "inner"}),
            }
        }

    RETURN_TYPES = (LAYER_GRADIENT,)
    FUNCTION = "gradient"
    CATEGORY = GRADIENT_CATEGORY
    
    def gradient(self, gradient_a : Dict[str, float], gradient_b : Dict[str, float], operation : str, **kwargs) -> Tuple[Dict[str, float]]:
        """
        Performs an operation on two gradients.
        
        Args:
            gradient_a (Dict[str, float]): The first gradient.
            gradient_b (Dict[str, float]): The second gradient.
            operation (str): The operation to perform.  One of "mean", "min", "max", "add", "subtract", "multiply", "divide".
            
        Returns:
            Tuple[Dict[str, float]]: A dictionary of layer gradients.
        """
        
        gradient : Dict[str, float] = {}
        keys = set(list(gradient_a.keys())).union(set(list(gradient_b.keys())))
        for k in keys:
            if k not in gradient_a.keys():
                if kwargs.get("join", "inner") == "inner":
                    continue
                else:
                    gradient[k] = gradient_b[k]
                    continue
            if k not in gradient_b.keys():
                if kwargs.get("join", "inner") == "inner":
                    continue
                else:
                    gradient[k] = gradient_a[k]
                    continue

            if operation == "mean":
                gradient[k] = (gradient_a[k] + gradient_b[k]) / 2.0
            elif operation == "min":
                gradient[k] = min(gradient_a[k], gradient_b[k])
            elif operation == "max":
                gradient[k] = max(gradient_a[k], gradient_b[k])
            elif operation == "add":
                gradient[k] = gradient_a[k] + gradient_b[k]
            elif operation == "subtract":
                gradient[k] = gradient_a[k] - gradient_b[k]
            elif operation == "multiply":
                gradient[k] = gradient_a[k] * gradient_b[k]
            elif operation == "divide":
                gradient[k] = gradient_a[k] / gradient_b[k]
            else:
                raise RuntimeError(f"Unknown operation {operation}")
        
        return (gradient, )


class MBWLayerGradient:
    """
    Define a gradient, like MBW.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the merging process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        arg_dict = {
                "model": ("MODEL",),
                "time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        argument = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
        for i in range(12):
            arg_dict[f"input_blocks.{i}"] = argument
        for i in range(3):
            arg_dict[f"middle_block.{i}"] = argument
        for i in range(12):
            arg_dict[f"output_blocks.{i}"] = argument
        arg_dict["out"] = argument
        opt = {}
        return {"required": arg_dict ,"optional": opt}

    RETURN_TYPES = (LAYER_GRADIENT,)
    FUNCTION = "gradient"
    CATEGORY = GRADIENT_CATEGORY
    
    def gradient(self, model : ModelPatcher, **kwargs) -> Tuple[Dict[str, float]]:
        """
        Returns a gradient dictionary for the specified model type.
        
        Args:
            model (ModelPatcher): The model to generate the gradient for.
            
        Returns:
            Tuple[Dict[str, float]]: A dictionary of layer gradients.
        """

        sd : Dict[str, torch.Tensor] = model.model_state_dict()
        gradient : Dict[str, float] = {}
        for k in sd.keys():
            ratio = self.calculate_layer_ratio(key=k, **kwargs)
            if ratio is None:
                continue
            gradient[k] = ratio

        return (gradient, )
        

    @classmethod
    def scan_layer(cls, key : str, base : str, n : int, **kwargs):
        for i in range(n):
            my_key = f"{base}.{i}"
            if key.startswith(my_key):
                if my_key in kwargs:
                    return kwargs[my_key]
                else:
                    print(f"No weight for {my_key}")
                    return None
            elif i==(n-1):
                print(f"Unknown key: {key},i={i}")
                return None
        return None

    @classmethod
    def calculate_layer_ratio(cls, key, **kwargs) -> Optional[float]:
        k_unet = key[len("diffusion_model."):]
        ratio = None

        # Get our ratio for this layer
        if k_unet.startswith(f"input_blocks."):
            # use scan layer static
            ratio = cls.scan_layer(k_unet, "input_blocks", 12, **kwargs)
        elif k_unet.startswith(f"middle_block."):
            ratio = cls.scan_layer(k_unet, "middle_block", 3, **kwargs)
        elif k_unet.startswith(f"output_blocks."):
            ratio = cls.scan_layer(k_unet, "output_blocks", 12, **kwargs)
        elif k_unet.startswith("out."):
            ratio = kwargs.get("out", None)
        elif k_unet.startswith("time"):
            ratio = kwargs.get("time", None)
        elif k_unet.startswith("label_emb"):
            ratio  = kwargs.get("label", None)
        else:
            print(f"Unknown key: {key}, skipping.")

        return ratio


class BlockLayerGradient:
    """
    Calculate a gradient, with a block structure.
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
                "model": ("MODEL",),
                "time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (LAYER_GRADIENT,)
    FUNCTION = "gradient"
    CATEGORY = GRADIENT_CATEGORY
    
    def gradient(self, model : ModelPatcher, **kwargs) -> Tuple[Dict[str, float]]:
        """
        Returns a gradient dictionary for the specified model type.
        
        Args:
            model (ModelPatcher): The model to generate the gradient for.
            time (float): The gradient for the time layers.
            label (float): The gradient for the label layers.
            input (float): The gradient for the input layers.
            middle (float): The gradient for the middle layers.
            output (float): The gradient for the output layers.
            out (float): The gradient for the out layers.
            
        Returns:
            Tuple[Dict[str, float]]: A dictionary of layer gradients.
        """
        sd : Dict[str, torch.Tensor] = model.model_state_dict()
        gradient : Dict[str, float] = {}
        for k in sd.keys():
            ratio = self.calculate_layer_ratio(key=k, **kwargs)
            if ratio is None:
                continue
            gradient[k] = ratio

        return (gradient, )
    
    @classmethod
    def calculate_layer_ratio(cls, key : str,
                              time : float, label : float, input : float, middle : float, output : float, out : float,
                              **kwargs) -> Optional[float]:
        k_unet = key[len("diffusion_model."):]
        ratio = None

        # Get our ratio for this layer
        if k_unet.startswith(f"input_blocks."):
            # use scan layer static
            ratio = input
        elif k_unet.startswith(f"middle_block."):
            ratio = middle
        elif k_unet.startswith(f"output_blocks."):
            ratio = output
        elif k_unet.startswith("out."):
            ratio = out
        elif k_unet.startswith("time"):
            ratio = time
        elif k_unet.startswith("label_emb"):
            ratio  = label
        else:
            print(f"Unknown key: {key}, skipping.")

        return ratio


class LayerGradientEdit:
    """
    Takes a layer gradient and edits it.
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
                "operation": (["set", "add", "subtract", "multiply", "divide"], {'default': "set"}),
                "value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "layers": ("STRING", {"multiline": True}),
            },
        }
        
    RETURN_TYPES = (LAYER_GRADIENT,)
    FUNCTION = "gradient"
    CATEGORY = GRADIENT_CATEGORY

    def gradient(self, gradient : Dict[str, float], operation : str, value : float, layers : str, **kwargs) -> Tuple[Dict[str, float]]:
        """
        Edits a layer gradient.
        
        Args:
            gradient (Dict[str, float]): The gradient.
            operation (str): The operation to perform.
            value (float): The value to use for the operation.
            layers (str): The layer names.  This can be a comma separated list, with wildcards, or using {0, 1} to target specific layers.
            
        Returns:
            Tuple[Dict[str, float]]: A dictionary of layer gradients.
        """
        keys = list(gradient.keys())
        collected_targets = collect_layers(layers, keys)
        if len(collected_targets) == 0:
            raise ValueError("No layers specified")

        for target in collected_targets:
            for layer in layers_for_mask(target, keys):
                #print(f"Editing layer {layer} with operation {operation} and value {value}")
                if operation == "set":
                    gradient[layer] = value
                elif operation == "add":
                    gradient[layer] += value
                elif operation == "subtract":
                    gradient[layer] -= value
                elif operation == "multiply":
                    gradient[layer] *= value
                elif operation == "divide":
                    gradient[layer] /= value
                else:
                    raise ValueError("Unknown operation: {}".format(operation))
            
        return (gradient,)
    