from .components.clip import DareClipMerger
from .components.dare import DareUnetMerger, DareUnetMergerGradient
from .components.dare_mbw import DareUnetMergerMBW
from .components.dare_element import DareUnetMergerElement
from .components.gradients import BlockLayerGradient, ShellLayerGradient, AttentionLayerGradient, MBWLayerGradient, LayerGradientOperations, LayerGradientEdit
from .components.block import BlockUnetMerger, GradientUnetMerger
from .components.normalize import NormalizeUnet
from .components.mask_model import MagnitudeMasker, MaskOperations, MaskEdit, SimpleMasker, QuadMasker
from .components.reports import MaskReporting, ModelReporting, LoRAReporting, LayerGradientReporting
from .components.lora import LoraLoaderTags


NODE_CLASS_MAPPINGS = {
    "DM_AdvancedModelMerger": GradientUnetMerger,
    "DM_AdvancedDareModelMerger": DareUnetMergerGradient,
    "DM_BlockModelMerger": BlockUnetMerger,
    "DM_DareModelMergerBlock": DareUnetMerger,
    "DM_DareModelMergerMBW": DareUnetMergerMBW,
    "DM_DareModelMergerElement": DareUnetMergerElement,
    "DM_DareClipMerger": DareClipMerger,
    "DM_SimpleMasker": SimpleMasker,
    "DM_MagnitudeMasker": MagnitudeMasker,
    "DM_QuadMasker": QuadMasker,
    "DM_MaskOperations": MaskOperations,
    "DM_MaskEdit": MaskEdit,
    "DM_GradientOperations": LayerGradientOperations,
    "DM_GradientEdit": LayerGradientEdit,
    "DM_BlockGradient": BlockLayerGradient,
    "DM_ShellGradient": ShellLayerGradient,
    "DM_AttentionGradient": AttentionLayerGradient,
    "DM_MBWGradient": MBWLayerGradient,
    "DM_ModelReporting": ModelReporting,
    "DM_MaskReporting": MaskReporting,
    "DM_LoRAReporting": LoRAReporting,
    "DM_GradientReporting": LayerGradientReporting,
    "DM_NormalizeModel": NormalizeUnet,
    "DM_LoRALoaderTags": LoraLoaderTags,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DM_AdvancedModelMerger": "Model Merger (Advanced)",
    "DM_AdvancedDareModelMerger": "Model Merger (Advanced/DARE)",
    "DM_BlockModelMerger": "Model Merger (Block)",
    "DM_DareModelMergerBlock": "Model Merger (Block/DARE)",
    "DM_DareModelMergerMBW": "Model Merger (MBW/DARE)",
    "DM_DareModelMergerElement": "Model Merger (Attention/DARE)",
    "DM_DareClipMerger": "CLIP Merger (DARE)",
    "DM_SimpleMasker": "Simple Masker",
    "DM_MagnitudeMasker": "Magnitude Masker",
    "DM_QuadMasker": "Quad Masker",
    "DM_MaskOperations": "Mask Operations",
    "DM_MaskEdit": "Mask Edit",
    "DM_GradientOperations": "Gradient Operations",
    "DM_GradientEdit": "Gradient Edit",
    "DM_BlockGradient": "Block Gradient",
    "DM_ShellGradient": "Shell Gradient",
    "DM_AttentionGradient": "Attention Gradient",
    "DM_MBWGradient": "MBW Gradient",
    "DM_ModelReporting": "Model Reporting",
    "DM_MaskReporting": "Mask Reporting",
    "DM_LoRAReporting": "LoRA Reporting",
    "DM_GradientReporting": "Gradient Reporting",
    "DM_NormalizeModel": "Normalize Model",
    "DM_LoRALoaderTags": "LoRA Loader (Tags)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
