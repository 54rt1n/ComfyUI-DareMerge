# ddare/mask.py

import torch
from typing import Dict, Optional


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
        return self.state_dict[layer_name]

    def model_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.state_dict

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
        