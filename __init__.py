from .components.clip import DareClipMerger
from .components.dare import DareUnetMerger
from .components.dare_mbw import DareUnetMergerMBW
from .components.block import BlockUnetMerger
from .components.normalize import NormalizeUnet
from .components.mask_model import MagnitudeMasker, MaskOperations, MaskEdit, SimpleMasker, QuadMasker
from .components.reports import MaskReporting, ModelReporting


NODE_CLASS_MAPPINGS = {
    "DM_MaskedModelMerger": BlockUnetMerger,
    "DM_DareModelMerger": DareUnetMerger,
    "DM_DareModelMergerMBW": DareUnetMergerMBW,
    "DM_DareClipMerger": DareClipMerger,
    "DM_SimpleMasker": SimpleMasker,
    "DM_MagnitudeMasker": MagnitudeMasker,
    "DM_QuadMasker": QuadMasker,
    "DM_MaskOperations": MaskOperations,
    "DM_MaskEdit": MaskEdit,
    "DM_ModelReporting": ModelReporting,
    "DM_MaskReporting": MaskReporting,
    "DM_NormalizeModel": NormalizeUnet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DM_MaskedModelMerger": "Model Merger (Masked)",
    "DM_DareModelMerger": "Model Merger (DARE)",
    "DM_DareModelMergerMBW": "MBW Merger (DARE)",
    "DM_DareClipMerger": "CLIP Merger (DARE)",
    "DM_SimpleMasker": "Simple Masker",
    "DM_MagnitudeMasker": "Magnitude Masker",
    "DM_QuadMasker": "Quad Masker",
    "DM_MaskOperations": "Mask Operations",
    "DM_MaskEdit": "Mask Edit",
    "DM_ModelReporting": "Model Reporting",
    "DM_MaskReporting": "Mask Reporting",
    "DM_NormalizeModel": "Normalize Model",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
