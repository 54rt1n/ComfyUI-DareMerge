from .merge.dare import DareModelMerger
from .merge.dareMBW import DareModelMergerMBW
from .merge.mag import MagnitudePruningModelMerger
from .merge.block import BlockModelMergerAdv


NODE_CLASS_MAPPINGS = {
    "DareModelMerger": DareModelMerger,
    "DareModelMergerMBW": DareModelMergerMBW,
    "MagnitudeModelMerger": MagnitudePruningModelMerger,
    "BlockModelMergerAdv": BlockModelMergerAdv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DareModelMerger": "ModelMergeByDARE",
    "DareModelMergerMBW": "ModelMergeByDAREMBW",
    "MagnitudeModelMerger": "ModelMergeByMagnitudePruning",
    "BlockModelMergerAdv": "ModelMergeByBlock (Advanced)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
