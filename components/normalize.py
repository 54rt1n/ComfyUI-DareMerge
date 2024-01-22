# components/normalize.py
import torch
from typing import Dict, Tuple

from comfy.model_patcher import ModelPatcher
from ..ddare.util import cuda_memory_profiler, get_device
from ..ddare.tensor import relative_norm
from ..ddare.const import UTIL_CATEGORY

"""
These are the layers that we are going to normalize, and how we are going to normalize them:
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.0.in_layers.{0,2}.{weight, bias} scale by weight
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.0.emb_layers.{1}.{weight, bias} scale by weight
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.0.out_layers.{0,3}.{weight, bias} scale by weight
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.1.{norm, proj_in, proj_out}.{weight, bias} scale by weight
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.1.transformer_blocks.0.{attn1, attn2}.{to_q, to_k, to_v}.weight q, v scaled by q weight, k inverse scaled
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.1.transformer_blocks.0.{attn1, attn2}.to_out.0.{weight, bias} scale by weight
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.1.transformer_blocks.0.ff.net.{0,2}.proj.{weight, bias} scale by weight
diffusion_model.{{input_blocks, output_blocks}.{n}, {middle_block}}.1.transformer_blocks.0.{norm1, norm2, norm3}.{weight, bias} scale by weight
"""

class NormalizeUnet:
    """
    A class to normalize the blocks from one model to the other, bringing them into the same scale.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:
        """
        Defines the input types for the merging process.

        Returns:
            Dict[str, tuple]: A dictionary specifying the required model types and parameters.
        """
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "method": (["q_norm", "all", "none", "attn_only"], {"default": "attn_only"}),
                "magnify": (["off", "on"], {"default": "off"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "normalize"
    CATEGORY = UTIL_CATEGORY

    def generate_key_groups_sd15(self):
        # Lets build our key selection strategy, and pick out the keys we want to patch from our key list
        # This is currently pretty manual, but we can make it more automatic later.  Right now we are just
        # making sure it matches sd15.
        blocks = []
        for i in range(12):
            blocks.append(f"input_blocks.{i}")
            blocks.append(f"output_blocks.{i}")
        blocks.append("middle_block")
    
        key_groups = {}
        prefix = "diffusion_model"
        layers = [("0.in_layers", (0,2)), ("0.emb_layers", (1,)), ("0.out_layers", (0,3))]
        layers += [("2.in_layers", (0,2)), ("2.emb_layers", (1,)), ("2.out_layers", (0,3))]
        layers += [("1.norm", ()), ("1.proj_in", ()), ("1.proj_out", ())]
        for b in blocks:
            for lk, z in layers:
                if len(z) == 0:
                    key_groups[f"{prefix}.{b}.{lk}.weight"] = (f"{prefix}.{b}.{lk}.weight", f"{prefix}.{b}.{lk}.bias")
                else:
                    for i in z:
                        key_groups[f"{prefix}.{b}.{lk}.{i}.weight"] = (f"{prefix}.{b}.{lk}.{i}.weight", f"{prefix}.{b}.{lk}.{i}.bias")
            key_groups[f"{prefix}.{b}.0.skip_connection.weight"] = (f"{prefix}.{b}.0.skip_connection.weight", f"{prefix}.{b}.0.skip_connection.bias")
            key_groups[f"{prefix}.{b}.1.transformer_blocks.0.norm1.weight"] = (f"{prefix}.{b}.1.transformer_blocks.0.norm1.weight", f"{prefix}.{b}.1.transformer_blocks.0.norm1.bias")
            key_groups[f"{prefix}.{b}.1.transformer_blocks.0.norm2.weight"] = (f"{prefix}.{b}.1.transformer_blocks.0.norm2.weight", f"{prefix}.{b}.1.transformer_blocks.0.norm2.bias")
            key_groups[f"{prefix}.{b}.1.transformer_blocks.0.norm3.weight"] = (f"{prefix}.{b}.1.transformer_blocks.0.norm3.weight", f"{prefix}.{b}.1.transformer_blocks.0.norm3.bias")
            key_groups[f"{prefix}.{b}.1.transformer_blocks.0.ff.net.0.proj.weight"] = (f"{prefix}.{b}.1.transformer_blocks.0.ff.net.0.proj.weight", f"{prefix}.{b}.1.transformer_blocks.0.ff.net.0.proj.bias")
            key_groups[f"{prefix}.{b}.1.transformer_blocks.0.ff.net.2.weight"] = (f"{prefix}.{b}.1.transformer_blocks.0.ff.net.2.weight", f"{prefix}.{b}.1.transformer_blocks.0.ff.net.2.bias")
            # Get our two attention blocks into our 5-tuple
            key_groups[f"{prefix}.{b}.1.transformer_blocks.0.attn1.to_q.weight"] = (f"{prefix}.{b}.1.transformer_blocks.0.attn1.to_q.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn1.to_k.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn1.to_v.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn1.to_out.0.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn1.to_out.0.bias")
            key_groups[f"{prefix}.{b}.1.transformer_blocks.0.attn2.to_q.weight"] = (f"{prefix}.{b}.1.transformer_blocks.0.attn2.to_q.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn2.to_k.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn2.to_v.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn2.to_out.0.weight", f"{prefix}.{b}.1.transformer_blocks.0.attn2.to_out.0.bias")
            key_groups[f"{prefix}.{b}.2.conv.weight"] = (f"{prefix}.{b}.2.conv.weight", f"{prefix}.{b}.2.conv.bias")
        key_groups[f"{prefix}.input_blocks.0.weight"] = (f"{prefix}.input_blocks.0.weight", f"{prefix}.input_blocks.0.bias") 
        key_groups[f"{prefix}.input_blocks.0.0.weight"] =  (f"{prefix}.input_blocks.0.0.weight", f"{prefix}.input_blocks.0.0.bias")
        key_groups[f"{prefix}.input_blocks.3.0.op.weight"] = (f"{prefix}.input_blocks.3.0.op.weight", f"{prefix}.input_blocks.3.0.op.bias")
        key_groups[f"{prefix}.input_blocks.6.0.op.weight"] = (f"{prefix}.input_blocks.6.0.op.weight", f"{prefix}.input_blocks.6.0.op.bias")
        key_groups[f"{prefix}.input_blocks.9.0.op.weight"] = (f"{prefix}.input_blocks.9.0.op.weight", f"{prefix}.input_blocks.9.0.op.bias")
        key_groups[f"{prefix}.out.0.weight"] = (f"{prefix}.out.0.weight", f"{prefix}.out.0.bias")
        key_groups[f"{prefix}.out.2.weight"] = (f"{prefix}.out.2.weight", f"{prefix}.out.2.bias")
        key_groups[f"{prefix}.output_blocks.2.1.conv.weight"] = (f"{prefix}.output_blocks.2.1.conv.weight", f"{prefix}.output_blocks.2.1.conv.bias")
        key_groups[f"{prefix}.time_embed.0.weight"] = (f"{prefix}.time_embed.0.weight", f"{prefix}.time_embed.0.bias")
        key_groups[f"{prefix}.time_embed.2.weight"] = (f"{prefix}.time_embed.2.weight", f"{prefix}.time_embed.2.bias")

        return key_groups

    def normalize(self, model_a: ModelPatcher, model_b: ModelPatcher, method : str, magnify : str = "off", **kwargs) -> Tuple[ModelPatcher]:
        """
        Scales model A by the scaling factor calculated from model B.

        Args:
            model_a (ModelPatcher): Model to be scaled.
            model_b (ModelPatcher): Model to be used as reference.
            method (str): Method to be used for merging.
            **kwargs: Additional arguments specifying the merge ratios for different layers and sparsity.

        Returns:
            Tuple[ModelPatcher]: A tuple containing the merged ModelPatcher instance.
        """

        device = get_device()

        scaler = lambda a,b: relative_norm(a,b)
        if magnify == "on":
            scaler = lambda a,b: relative_norm(b,a)

        with cuda_memory_profiler():
            m = model_a.clone()  # Clone model_a to keep its structure
            if method == "none":
                if len(m.patches) > 0:
                    print(f"Model A has patches: {m.patches.keys()}")
                return (m,)
            if len(model_a.patches) > 0:
                print("Model A has patches, applying them")
                m.patch_model(None, True)
                model_a_sd = m.model_state_dict()  # State dict of model_a
                m.unpatch_model()  # Unpatch model_a
            else:
                model_a_sd = m.model_state_dict()  # State dict of model_a

            if len(model_b.patches) > 0:
                print("Model B has patches, applying them")
                model_b.patch_model(None, True)
                model_b_sd = model_b.model_state_dict()
                model_b.unpatch_model()
            else:
                model_b_sd = model_b.model_state_dict()

            strength_patch = 1.0
            strength_model = 0.0

            processed_keys = {}
            for k in sorted(model_a_sd.keys()):
                processed_keys[k] = False

            for key, group in self.generate_key_groups_sd15().items():
                if key not in model_a_sd or key not in model_b_sd:
                    #print("could not patch. key doesn't exist in model:", key)
                    continue

                elif len(group) == 2 and (method != "attn_only"):
                    # Normalize our weight and bias
                    weight_key, bias_key = group
                    weight_a : torch.Tensor = model_a_sd[weight_key].to(device)
                    weight_b : torch.Tensor = model_b_sd[weight_key].to(device)
                    bias_a : torch.Tensor = model_a_sd[bias_key].to(device)

                    scale = scaler(weight_a, weight_b).to(device)
                    na = torch.empty_like(weight_a, device=device)
                    na = weight_a * scale
                    nb = torch.empty_like(weight_b, device=device)
                    nb = weight_b / scale
                    
                    #print("normalized:", weight_key, scale)
                    del scale

                    m.add_patches({weight_key: (na.to('cpu'),)}, strength_patch, strength_model)
                    m.add_patches({bias_key: (nb.to('cpu'),)}, strength_patch, strength_model)
                    
                    weight_a.to("cpu")
                    weight_b.to("cpu")
                    bias_a.to("cpu")

                    processed_keys[weight_key] = True
                    processed_keys[bias_key] = True
                elif len(group) == 5 and (method == "q_norm" or method == "attn_only"):
                    # Scaled attention, we determine the scaling factor from the q weight
                    q, k, v, out_w, out_b = group
                    q_a : torch.Tensor = model_a_sd[q].to(device)
                    q_b : torch.Tensor = model_b_sd[q].to(device)
                    k_a : torch.Tensor = model_a_sd[k].to(device)
                    #v_a : torch.Tensor = model_a_sd[v].to(device)
                    out_w_a : torch.Tensor = model_a_sd[out_w].to(device)
                    out_w_b : torch.Tensor = model_b_sd[out_w].to(device)
                    out_b_a : torch.Tensor = model_a_sd[out_b].to(device)
                    scale_a = scaler(q_a, q_b).to(device)
                    # allocate q, k, v, out_w, out_b
                    nq = torch.empty_like(q_a, device=device)
                    nq = q_a.to(device) * scale_a
                    nk = torch.empty_like(k_a, device=device)
                    nk = k_a.to(device) / scale_a
                    #v_a = v_a.copy_(v_a * scale).to("cpu")
                    scale_o = scaler(out_w_a, out_w_b)
                    nout_w = torch.empty_like(out_w_a, device=device)
                    nout_w = out_w_a.to(device) * scale_o
                    nout_b = torch.empty_like(out_b_a, device=device)
                    nout_b = out_b_a.to(device) * scale_o

                    #print("normalized:", q, scale_a, scale_o)
                    del scale_a, scale_o

                    m.add_patches({q: (nq.to('cpu'),)}, strength_patch, strength_model)
                    m.add_patches({k: (nk.to('cpu'),)}, strength_patch, strength_model)
                    #m.add_patches({v: (v_a,)}, strength_patch, strength_model)
                    m.add_patches({out_w: (nout_w.to('cpu'),)}, strength_patch, strength_model)
                    m.add_patches({out_b: (nout_b.to('cpu'),)}, strength_patch, strength_model)
                    
                    q_a.to("cpu")
                    q_b.to("cpu")
                    k_a.to("cpu")
                    #v_a.to("cpu")
                    out_w_a.to("cpu")
                    out_w_b.to("cpu")
                    out_b_a.to("cpu")

                    processed_keys[q] = True
                    processed_keys[k] = True
                    processed_keys[v] = True
                    processed_keys[out_w] = True
                    processed_keys[out_b] = True
                
        for k, v in processed_keys.items():
            if not v and method == "q_norm":
                #print("key not processed:", k)
                pass

        return (m,)
