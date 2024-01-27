# components/lora.py
import folder_paths
from comfy.sd import load_lora_for_models
from comfy.sd import CLIP
from comfy.model_patcher import ModelPatcher

from ..ddare.lora import DoctorLora
from ..ddare.const import LORA_CATEGORY

class LoraLoaderTags:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                              "tag_limit": ("INT", {"default": 10, "min": 1, "max": 100}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"

    CATEGORY = LORA_CATEGORY

    def load_lora(self, model : ModelPatcher, clip : CLIP, lora_name : str, strength_model : float, strength_clip : float, tag_limit : int = 10, **kwargs):
        tags = ""
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, tags)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            drlora = DoctorLora.load(lora_path)

            tags = drlora.tags
            if tags is not None:
                tags = ", ".join(tags[:tag_limit])

            lora = drlora.lora

        model_lora, clip_lora = load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora, tags)
 