# ddare/mask.py
import torch
from typing import Dict, Optional

from .tensor import bernoulli_noise, gaussian_noise

# This is very memory heavy.  Making true/false or sparse masks would be better; and patterns such as triangle, or right/left, up/down split.
class ModelMask:
    """ 
    A container to hold a state dict of masks for a model.
    """
    def __init__(self, state_dict : Dict[str, torch.Tensor]):
        self.state_dict = state_dict

    def add_layer_mask(self, layer_name : str, mask : torch.Tensor):
        self.state_dict[layer_name] = mask.clone().to("cpu")

    def get_layer_mask(self, layer_name : str) -> Optional[torch.Tensor]:
        if layer_name not in self.state_dict:
            return None
        return self.state_dict[layer_name].bool()

    def model_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.state_dict

    def clone(self) -> 'ModelMask':
        return ModelMask({k: v.clone().to("cpu") for k, v in self.state_dict.items()})

    def noise_layer(self, layer_name : str, noise_type : str, v0 : float, v1 : float, **kwargs):
        """
        Takes our current mask and adds noise to it.
        """
        layer = self.state_dict[layer_name]
        if noise_type == "bernoulli":
            noise = bernoulli_noise(layer, v0)
        elif noise_type == "gaussian":
            noise = gaussian_noise(layer, v0, v1)
        else:
            raise ValueError(f"Unknown noise type {noise_type}")
        
        del self.state_dict[layer_name]
        
        self.state_dict[layer_name] = noise.to("cpu")

    def boolean_layer(self, layer_name : str, direction : bool, **kwargs):
        """
        Takes our current mask and adds noise to it.
        """
        layer = self.state_dict[layer_name]
        if direction:
            tensor = torch.ones(layer.shape)
        else:
            tensor = torch.zeros(layer.shape)
        
        del self.state_dict[layer_name]
        
        self.state_dict[layer_name] = tensor.to("cpu")
        
    @classmethod
    def union(cls, mask_a : 'ModelMask', mask_b : 'ModelMask') -> 'ModelMask':
        """
        Take two masks and perform a union operation.

        Args:
            mask_a (ModelMask): The first mask.
            mask_b (ModelMask): The second mask.

        Returns:
            ModelMask: A new mask containing the union of the two masks.
        """
        new_state_dict = {}
        for k in mask_a.state_dict.keys():
            if k not in mask_b.state_dict:
                new_state_dict[k] = mask_a.state_dict[k].clone().to("cpu")
                continue

            a = mask_a.state_dict[k]
            b = mask_b.state_dict[k]

            new_state_dict[k] = torch.logical_or(a, b).clone().to("cpu")

        return ModelMask(new_state_dict)

    @classmethod
    def intersect(cls, mask_a : 'ModelMask', mask_b : 'ModelMask') -> 'ModelMask':
        """
        Take two masks and perform an intersection operation.

        Args:
            mask_a (ModelMask): The first mask.
            mask_b (ModelMask): The second mask.

        Returns:
            ModelMask: A new mask containing the intersection of the two masks.
        """
        new_state_dict = {}
        for k in mask_a.state_dict.keys():
            if k not in mask_b.state_dict:
                continue

            a = mask_a.state_dict[k]
            b = mask_b.state_dict[k]

            new_state_dict[k] = torch.logical_and(a, b).clone().to("cpu")

        return ModelMask(new_state_dict)

    @classmethod
    def set_difference(cls, mask_a : 'ModelMask', mask_b : 'ModelMask') -> 'ModelMask':
        """
        Take two masks and perform a set difference operation.

        Args:
            mask_a (ModelMask): The first mask.
            mask_b (ModelMask): The second mask.

        Returns:
            ModelMask: A new mask containing the set difference of the two masks.
        """
        new_state_dict = {}
        for k in mask_a.state_dict.keys():
            if k not in mask_b.state_dict:
                new_state_dict[k] = mask_a.state_dict[k].clone().to("cpu")
                continue

            a = mask_a.state_dict[k]
            b = mask_b.state_dict[k]

            new_state_dict[k] = torch.logical_and(a, torch.logical_not(b)).clone().to("cpu")

        return ModelMask(new_state_dict)
    
    @classmethod
    def symmetric_distance(cls, mask_a: 'ModelMask', mask_b: 'ModelMask') -> 'ModelMask':
        """
        Take two masks and perform a symmetric distance operation.

        Args:
            mask_a (ModelMask): The first mask.
            mask_b (ModelMask): The second mask.

        Returns:
            ModelMask: A new mask containing the symmetric distance of the two masks.
        """
        new_state_dict = {}
        all_keys = set(mask_a.state_dict.keys()).union(set(mask_b.state_dict.keys()))

        for k in all_keys:
            a = mask_a.state_dict.get(k, None)
            b = mask_b.state_dict.get(k, None)

            if a is None:
                new_state_dict[k] = b.clone().to("cpu")
            elif b is None:
                new_state_dict[k] = a.clone().to("cpu")
            else:
                new_state_dict[k] = torch.logical_xor(a, b).clone().to("cpu")

        return ModelMask(new_state_dict)
        
