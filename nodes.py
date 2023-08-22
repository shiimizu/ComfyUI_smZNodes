from .modules import prompt_parser, devices, shared
from .modules.shared import opts
from .modules.sd_hijack import model_hijack
from .smZNodes import encode_from_texts, expand, run, LazyCond
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
from nodes import CLIPTextEncode, MAX_RESOLUTION
import comfy.sd
import comfy.model_management
import torch
import comfy.samplers
import comfy.sample
from functools import partial

class smZ_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP", ),
                "parser": (["comfy", "comfy++", "A1111", "full", "compel", "fixed attention"],{"default": "comfy"}),
                # whether weights are normalized by taking the mean
                "mean_normalization": ("BOOLEAN", {"default": True}),
                "multi_conditioning": ("BOOLEAN", {"default": True}),
                "use_old_emphasis_implementation": ("BOOLEAN", {"default": False}),
                "use_CFGDenoiser": ("BOOLEAN", {"default": False}),
                "with_SDXL": ("BOOLEAN", {"default": False}),
                "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "text_g": ("STRING", {"multiline": True, "placeholder": "CLIP_G"}), "clip": ("CLIP", ),
                "text_l": ("STRING", {"multiline": True, "placeholder": "CLIP_L"}), "clip": ("CLIP", ),
            },
            "hidden": {
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip: comfy.sd.CLIP, text, parser, mean_normalization,
               multi_conditioning, use_old_emphasis_implementation,
               use_CFGDenoiser, with_SDXL, ascore, width, height, crop_w, 
               crop_h, target_width, target_height, text_g, text_l):
        params = locals()
        devices.device = clip.patcher.load_device
        shared.device = devices.device
        is_sdxl = "SDXL" in type(clip.cond_stage_model).__name__

        dtype = torch.float16 if comfy.model_management.should_use_fp16(device=devices.device) else torch.float32
        devices.dtype_unet = torch.float16 if is_sdxl and not comfy.model_management.FORCE_FP32 else (_dtype if (_dtype:=clip.patcher.model_dtype() != None) else dtype)
        devices.unet_needs_upcast = shared.opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16
        devices.dtype = devices.dtype_unet
        devices.dtype_vae = comfy.model_management.vae_dtype()
        opts.conds_cache.clear()
        opts.conds_cache = {"positive":{}, "negative":{}}
        params.pop('self', None)
        result = run(**params)
        # result[0][0][1]['encode_fn'] = run
        result[0][0][1]['params'] = {}
        result[0][0][1]['params'].update(params)
        return (LazyCond(result[0]),)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "smZ CLIPTextEncode": smZ_CLIPTextEncode,
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "smZ CLIPTextEncode" : "CLIP Text Encode++",
}