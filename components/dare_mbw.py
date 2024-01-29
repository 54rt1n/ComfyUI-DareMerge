# components/dare_mbw.py
from comfy.model_patcher import ModelPatcher
from typing import Dict, Tuple

from ..ddare.const import UNET_CATEGORY
from ..ddare.util import merge_input_types

from .dare import DareUnetMergerGradient
from .gradients import MBWLayerGradient


class DareUnetMergerMBW:
    """
    A class to merge two diffusion U-Net models using calculated deltas, sparsification,
    and a weighted consensus method. This is the DARE-TIES method, and allows for fine
    grained control over the merge ratios for each layer like MBW.
    
    https://arxiv.org/pdf/2311.03099.pdf
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the merging process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        merged = merge_input_types(DareUnetMergerGradient.INPUT_TYPES(), MBWLayerGradient.INPUT_TYPES())
        del merged["required"]["gradient"]
        del merged["required"]["model"]
        return merged

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = UNET_CATEGORY

    def merge(self, model_a : ModelPatcher, **kwargs,) -> Tuple[ModelPatcher]:
        """
        Merges two ModelPatcher instances based on the weighted consensus of their parameters and sparsity.

        Args:
            model_a (ModelPatcher): The base model to be merged.
            model_b (ModelPatcher): The model to merge into the base model.
            method (str): The method to use for merging, either "lerp", "slerp", or "gradient".
            seed (int): The random seed to use for the merge.
            clear_cache (bool): Whether to clear the CUDA cache after each chunk. Default is False.
            iterations (int): The number of iterations to perform the merge.  Default is 1.
            model_mask (Optional[ModelMask]): The model mask to use for protection of our model_a. Default is None.
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the merged ModelPatcher instance.
        """

        gradient = MBWLayerGradient().gradient(model=model_a, **kwargs)[0]
        return DareUnetMergerGradient().merge(model_a=model_a, gradient=gradient, **kwargs)
