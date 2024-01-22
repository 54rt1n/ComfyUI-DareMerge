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

