from .components.clip import DareClipMerger
from .components.dare import DareUnetMerger
from .components.dare_mbw import DareUnetMergerMBW
from .components.block import BlockUnetMerger
from .components.normalize import NormalizeUnet
from .components.mask_model import MagnitudeMasker, MaskOperations


NODE_CLASS_MAPPINGS = {
    "DM_MaskedModelMerger": BlockUnetMerger,
    "DM_DareModelMerger": DareUnetMerger,
    "DM_DareModelMergerMBW": DareUnetMergerMBW,
    "DM_DareClipMerger": DareClipMerger,
    "DM_NormalizeModel": NormalizeUnet,
    "DM_MagnitudeMasker": MagnitudeMasker,
    "DM_MaskOperations": MaskOperations,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DM_MaskedModelMerger": "Model Merger (Masked)",
    "DM_DareModelMerger": "Model Merger (DARE)",
    "DM_DareModelMergerMBW": "MBW Merger (DARE)",
    "DM_DareClipMerger": "CLIP Merger (DARE)",
    "DM_NormalizeModel": "Normalize Model",
    "DM_MagnitudeMasker": "Magnitude Masker",
    "DM_MaskOperations": "Mask Operations",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
