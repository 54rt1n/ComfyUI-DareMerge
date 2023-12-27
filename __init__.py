from .nodes import DareModelMerger

NODE_CLASS_MAPPINGS = {
    "DareModelMerger": DareModelMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DareModelMerger": "ModelMergeByDARE"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
