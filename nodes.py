import torch
import inspect
from pathlib import Path
from functools import partial
import os
from .modules import prompt_parser, devices, shared
from .modules.sd_hijack import model_hijack
from .smZNodes import run, prepare_noise
from nodes import MAX_RESOLUTION
import comfy.sd
import comfy.model_management
import comfy.samplers
import comfy.sample
import copy

BOOLEAN = [False, True]
try:
    cwd_path = Path(__file__).parent
    comfy_path = cwd_path.parent.parent
    widgets_path = os.path.join(comfy_path, "web", "scripts", "widgets.js")
    with open(widgets_path, encoding='utf8') as f:
        widgets_js = f.read()
    if 'BOOLEAN(' in widgets_js:
        BOOLEAN = "BOOLEAN"
    del widgets_js
except Exception as err:
    print("[smZNodes]:", err)

class smZ_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP", ),
                "parser": (["comfy", "comfy++", "A1111", "full", "compel", "fixed attention"],{"default": "comfy"}),
                # whether weights are normalized by taking the mean
                "mean_normalization": (BOOLEAN, {"default": True}),
                "multi_conditioning": (BOOLEAN, {"default": True}),
                "use_old_emphasis_implementation": (BOOLEAN, {"default": False}),
                "with_SDXL": (BOOLEAN, {"default": False}),
                "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "text_g": ("STRING", {"multiline": True, "placeholder": "CLIP_G"}), 
                "text_l": ("STRING", {"multiline": True, "placeholder": "CLIP_L"}),
            },
            "optional": {
                "smZ_steps": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip: comfy.sd.CLIP, text, parser, mean_normalization,
               multi_conditioning, use_old_emphasis_implementation,
               with_SDXL, ascore, width, height, crop_w, 
               crop_h, target_width, target_height, text_g, text_l, smZ_steps=1):
        params = locals()
        params['steps'] = params.pop('smZ_steps', smZ_steps)
        from .modules.shared import opts
        is_sdxl = "SDXL" in type(clip.cond_stage_model).__name__

        should_use_fp16_signature = inspect.signature(comfy.model_management.should_use_fp16)
        _p = should_use_fp16_signature.parameters
        devices.device = shared.device = clip.patcher.load_device if hasattr(clip.patcher, 'load_device') else clip.device
        if 'device' in _p and 'prioritize_performance' in _p:
            should_use_fp16 = partial(comfy.model_management.should_use_fp16, device=devices.device, prioritize_performance=False)
        elif 'device' in should_use_fp16_signature.parameters:
            should_use_fp16 = partial(comfy.model_management.should_use_fp16, device=devices.device)
        else:
            should_use_fp16 = comfy.model_management.should_use_fp16
        dtype = torch.float16 if should_use_fp16() else torch.float32
        dtype_unet= dtype
        devices.dtype = dtype
        #  devices.dtype_unet was hijacked so it will be the correct dtype by default
        if devices.dtype_unet == torch.float16:
            devices.dtype_unet = dtype_unet
        devices.unet_needs_upcast = opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16
        devices.dtype_vae = comfy.model_management.vae_dtype() if hasattr(comfy.model_management, 'vae_dtype') else torch.float32

        params.pop('self', None)
        result = run(**params)
        result[0][0][1]['params'] = {}
        result[0][0][1]['params'].update(params)
        if opts.pad_cond_uncond:
            text=params['text']
            with_SDXL=params['with_SDXL']
            params['text'] = ''
            params['with_SDXL'] = False
            empty = run(**params)[0]
            params['text'] = text
            params['with_SDXL'] = with_SDXL
            shared.sd_model.cond_stage_model_empty_prompt = empty[0][0]
        return result

# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# Our any instance wants to be a wildcard string
anytype = AnyType("*")

class smZ_Settings:
    @classmethod
    def INPUT_TYPES(s):
        from .modules.shared import opts
        return {"required": {
                                "clip": ("CLIP", ),
                                },
                "optional": {
                    "extra": ("STRING", {"multiline": True, "default": '{"show":true}'}),

                    "ㅤ"*1: ( "STRING", {"multiline": False, "default": "Stable Diffusion"}),
                    "info_comma_padding_backtrack": ("STRING", {"multiline": True, "default": "Prompt word wrap length limit\nin tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"}),
                    "Prompt word wrap length limit": ("INT", {"default": opts.comma_padding_backtrack, "min": 0, "max": 74, "step": 1}),
                    # "enable_emphasis": (BOOLEAN, {"default": opts.enable_emphasis}),
                    "info_RNG": ("STRING", {"multiline": True, "default": "Random number generator source.\nchanges seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"}),
                    "RNG": (["cpu", "gpu", "nv"],{"default": opts.randn_source}),
                    
                    "ㅤ"*2: ("STRING", {"multiline": False, "default": "Compute Settings"}),
                    "info_disable_nan_check": ("STRING", {"multiline": True, "default": "Disable NaN check in produced images/latent spaces. Only for CFGDenoiser."}),
                    "disable_nan_check": (BOOLEAN, {"default": opts.disable_nan_check}),

                    "ㅤ"*3: ("STRING", {"multiline": False, "default": "Sampler parameters"}),
                    "info_eta_noise_seed_delta": ("STRING", {"multiline": True, "default": "Eta noise seed delta\ndoes not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"}),
                    "ENSD": ("INT", {"default": opts.eta_noise_seed_delta, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                    "info_sgm_noise_multiplier": ("STRING", {"multiline": True, "default": "SGM noise multiplier\nmatch initial noise to official SDXL implementation - only useful for reproducing images\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818"}),
                    "sgm_noise_multiplier": (BOOLEAN, {"default": opts.sgm_noise_multiplier}),
                    "info_upcast_sampling": ("STRING", {"multiline": True, "default": "upcast sampling.\nNo effect with --force-fp32. Usually produces similar results to --force-fp32 with better performance while using less memory."}),
                    "upcast_sampling": (BOOLEAN, {"default": opts.upcast_sampling}),

                    "ㅤ"*4: ("STRING", {"multiline": False, "default": "Optimizations"}),
                    "info_NGMS": ("STRING", {"multiline": True, "default": "Negative Guidance minimum sigma\nskip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster. Only for CFGDenoiser.\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177"}),
                    "NGMS": ("FLOAT", {"default": opts.s_min_uncond, "min": 0.0, "max": 4.0, "step": 0.01}),
                    "info_pad_cond_uncond": ("STRING", {"multiline": True, "default": "Pad prompt/negative prompt to be same length\nimproves performance when prompt and negative prompt have different lengths; changes seeds. Only for CFGDenoiser."}),
                    "pad_cond_uncond": (BOOLEAN, {"default": opts.pad_cond_uncond}),
                    "info_batch_cond_uncond": ("STRING", {"multiline": True, "default": "Batch cond/uncond\ndo both conditional and unconditional denoising in one batch; uses a bit more VRAM during sampling, but improves speed – enabled on SDXL models. Only for CFGDenoiser."}),
                    "batch_cond_uncond": (BOOLEAN, {"default": opts.batch_cond_uncond}),

                    "ㅤ"*5: ("STRING", {"multiline": False, "default": "Compatibility"}),
                    "info_use_prev_scheduling": ("STRING", {"multiline": True, "default": "Previous prompt editing timelines\nFor [red:green:N]; previous: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"}),
                    "Use previous prompt editing timelines": (BOOLEAN, {"default": opts.use_old_scheduling}),
                    
                    "ㅤ"*6: ("STRING", {"multiline": False, "default": "Experimental"}),
                    "info_use_CFGDenoiser": ("STRING", {"multiline": True, "default": "CFGDenoiser\nAn experimental option to use stable-diffusion-webui's denoiser. It may not work as expected with inpainting/UnCLIP models or ComfyUI's Conditioning nodes, but it allows you to disable batch_cond_uncond and have different looking results."}),
                    "Use CFGDenoiser": (BOOLEAN, {"default": opts.use_CFGDenoiser}),
                    "info_debug": ("STRING", {"multiline": True, "default": "Debugging messages in the console."}),
                    "Debug": (BOOLEAN, {"default": opts.debug, "label_on": "on", "label_off": "off"}),
                }}
    RETURN_TYPES = ("CLIP",)
    OUTPUT_NODE = False
    FUNCTION = "run"
    CATEGORY = "advanced"

    def run(self, *args, **kwargs):
        from .modules.shared import opts as _opts
        device = comfy.model_management.get_torch_device()

        clip = kwargs.pop('clip', None) if 'clip' in kwargs else args[0]
        kwargs['s_min_uncond'] = max(min(kwargs.pop('NGMS'), 4.0), 0)
        kwargs['comma_padding_backtrack'] = kwargs.pop('Prompt word wrap length limit')
        kwargs['comma_padding_backtrack'] = max(min(kwargs['comma_padding_backtrack'], 74), 0)
        kwargs['use_old_scheduling']=kwargs.pop("Use previous prompt editing timelines")
        kwargs['use_CFGDenoiser'] = kwargs.pop("Use CFGDenoiser")
        kwargs['debug'] = kwargs.pop('Debug')
        kwargs['randn_source'] = kwargs.pop('RNG')
        kwargs['eta_noise_seed_delta'] = kwargs.pop('ENSD')
        
        opts = copy.deepcopy(_opts)
        [kwargs.pop(k, None) for k in [k for k in kwargs.keys() if 'info' in k or 'heading' in k or 'ㅤ' in k]]
        for k,v in kwargs.items():
            setattr(opts, k, v)

        clip = clip.clone()
        clip.patcher.model_options['smZ_opts'] = opts

        if not hasattr(comfy.sample, 'prepare_noise_orig'):
            comfy.sample.prepare_noise_orig = comfy.sample.prepare_noise
        if opts.randn_source == 'cpu':
            device = torch.device("cpu")
        _prepare_noise = partial(prepare_noise, device=device.type)
        comfy.sample.prepare_noise = _prepare_noise
        return (clip,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "smZ CLIPTextEncode": smZ_CLIPTextEncode,
    "smZ Settings": smZ_Settings,
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "smZ CLIPTextEncode" : "CLIP Text Encode++",
    "smZ Settings" : "Settings (smZ)",
}