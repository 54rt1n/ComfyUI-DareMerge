from .daremerge import DareModelMerger
from .magmerge import MagnitudePruningModelMerger

NODE_CLASS_MAPPINGS = {
    "DareModelMerger": DareModelMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DareModelMerger": "ModelMergeByDARE",
    "MagnitudeModelMerger": "ModelMergeByMagnitudePruning"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
