import torch
import inspect
from pathlib import Path
from functools import partial
import os
from .modules import prompt_parser, devices, shared
from .modules.sd_hijack import model_hijack
from .smZNodes import run, prepare_noise
from nodes import MAX_RESOLUTION
import comfy.model_patcher
import comfy.sd
import comfy.model_management
import comfy.samplers
import comfy.sample
from copy import deepcopy

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
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
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
                "text_g": ("STRING", {"multiline": True, "placeholder": "CLIP_G", "dynamicPrompts": True}), 
                "text_l": ("STRING", {"multiline": True, "placeholder": "CLIP_L", "dynamicPrompts": True}),
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
        from .modules.shared import opts_default as opts

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
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

# Our any instance wants to be a wildcard string
anytype = AnyType("*")

class smZ_Settings:
    @classmethod
    def INPUT_TYPES(s):
        from .modules.shared import opts_default as opts
        import json
        i = 0
        def create_heading():
            nonlocal i
            return "ㅤ"*(i:=i+1)
        create_heading_value = lambda x: ("STRING", {"multiline": False, "default": x, "placeholder": x})
        optional = {
            # "show_headings": (BOOLEAN, {"default": True}),
            # "show_descriptions": (BOOLEAN, {"default":True}),

            create_heading(): create_heading_value("Stable Diffusion"),
            "info_comma_padding_backtrack": ("STRING", {"multiline": True, "placeholder": "Prompt word wrap length limit\nin tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"}),
            "Prompt word wrap length limit": ("INT", {"default": opts.comma_padding_backtrack, "min": 0, "max": 74, "step": 1}),
            "enable_emphasis": (BOOLEAN, {"default": opts.enable_emphasis}),

            "info_RNG": ("STRING", {"multiline": True, "placeholder": "Random number generator source.\nchanges seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"}),
            "RNG": (["cpu", "gpu", "nv"],{"default": opts.randn_source}),
            
            create_heading(): create_heading_value("Compute Settings"),
            "info_disable_nan_check": ("STRING", {"multiline": True, "placeholder": "Disable NaN check in produced images/latent spaces. Only for CFGDenoiser."}),
            "disable_nan_check": (BOOLEAN, {"default": opts.disable_nan_check}),

            create_heading(): create_heading_value("Sampler parameters"),
            "info_eta_ancestral": ("STRING", {"multiline": True, "placeholder": "Eta for k-diffusion samplers\nnoise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"}),
            "eta": ("FLOAT", {"default": opts.eta, "min": 0.0, "max": 1.0, "step": 0.01}),
            "info_s_churn": ("STRING", {"multiline": True, "placeholder": "Sigma churn\namount of stochasticity; only applies to Euler, Heun, Heun++2, and DPM2"}),
            "s_churn": ("FLOAT", {"default": opts.s_churn, "min": 0.0, "max": 100.0, "step": 0.01}),
            "info_s_tmin": ("STRING", {"multiline": True, "placeholder": "Sigma tmin\nenable stochasticity; start value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2'"}),
            "s_tmin": ("FLOAT", {"default": opts.s_tmin, "min": 0.0, "max": 10.0, "step": 0.01}),
            "info_s_tmax": ("STRING", {"multiline": True, "placeholder": "Sigma tmax\n0 = inf; end value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2"}),
            "s_tmax": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "step": 0.01}),
            "info_s_noise": ("STRING", {"multiline": True, "placeholder": "Sigma noise\namount of additional noise to counteract loss of detail during sampling"}),
            "s_noise": ("FLOAT", {"default": opts.s_noise, "min": 0.0, "max": 1.1, "step": 0.001}),
            "info_eta_noise_seed_delta": ("STRING", {"multiline": True, "placeholder": "Eta noise seed delta\ndoes not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"}),
            "ENSD": ("INT", {"default": opts.eta_noise_seed_delta, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "info_sgm_noise_multiplier": ("STRING", {"multiline": True, "placeholder": "SGM noise multiplier\nmatch initial noise to official SDXL implementation - only useful for reproducing images\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818"}),
            "sgm_noise_multiplier": (BOOLEAN, {"default": opts.sgm_noise_multiplier}),
            "info_upcast_sampling": ("STRING", {"multiline": True, "placeholder": "upcast sampling.\nNo effect with --force-fp32. Usually produces similar results to --force-fp32 with better performance while using less memory."}),
            "upcast_sampling": (BOOLEAN, {"default": opts.upcast_sampling}),

            create_heading(): create_heading_value("Optimizations"),
            "info_NGMS": ("STRING", {"multiline": True, "placeholder": "Negative Guidance minimum sigma\nskip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster. Only for CFGDenoiser.\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177"}),
            "NGMS": ("FLOAT", {"default": opts.s_min_uncond, "min": 0.0, "max": 15.0, "step": 0.01}),
            "info_pad_cond_uncond": ("STRING", {"multiline": True, "placeholder": "Pad prompt/negative prompt to be same length\nimproves performance when prompt and negative prompt have different lengths; changes seeds. Only for CFGDenoiser."}),
            "pad_cond_uncond": (BOOLEAN, {"default": opts.pad_cond_uncond}),
            "info_batch_cond_uncond": ("STRING", {"multiline": True, "placeholder": "Batch cond/uncond\ndo both conditional and unconditional denoising in one batch; uses a bit more VRAM during sampling, but improves speed. Only for CFGDenoiser."}),
            "batch_cond_uncond": (BOOLEAN, {"default": opts.batch_cond_uncond}),

            create_heading(): create_heading_value("Compatibility"),
            "info_use_prev_scheduling": ("STRING", {"multiline": True, "placeholder": "Previous prompt editing timelines\nFor [red:green:N]; previous: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"}),
            "Use previous prompt editing timelines": (BOOLEAN, {"default": opts.use_old_scheduling}),
            
            create_heading(): create_heading_value("Experimental"),
            "info_use_CFGDenoiser": ("STRING", {"multiline": True, "placeholder": "CFGDenoiser\nAn experimental option to use stable-diffusion-webui's denoiser. It allows you to use the 'Optimizations' settings listed here."}),
            "Use CFGDenoiser": (BOOLEAN, {"default": opts.use_CFGDenoiser}),
            "info_debug": ("STRING", {"multiline": True, "placeholder": "Debugging messages in the console."}),
            "debug": (BOOLEAN, {"default": opts.debug, "label_on": "on", "label_off": "off"}),
        }
        # i = -1
        return {"required": {
                                "*": (anytype, {"forceInput": True}),
                                },
                "optional": {
                    # "extra": ("STRING", {"multiline": True, "default": json.dumps(optional)}),
                    "extra": ("STRING", {"multiline": True, "default": '{"show_headings":true,"show_descriptions":false,"mode":"*"}'}),
                    **optional
                }}
    RETURN_TYPES = (anytype,)
    FUNCTION = "run"
    CATEGORY = "advanced"

    def run(self, *args, **kwargs):
        first = kwargs.pop('*', None) if '*' in kwargs else args[0]
        if not hasattr(first, 'clone') or first is None: return (first,)

        kwargs['s_min_uncond'] = kwargs.pop('NGMS', 0.0)
        kwargs['comma_padding_backtrack'] = kwargs.pop('Prompt word wrap length limit')
        kwargs['use_old_scheduling']=kwargs.pop("Use previous prompt editing timelines")
        kwargs['use_CFGDenoiser'] = kwargs.pop("Use CFGDenoiser")
        kwargs['randn_source'] = kwargs.pop('RNG')
        kwargs['eta_noise_seed_delta'] = kwargs.pop('ENSD')
        kwargs['s_tmax'] = kwargs['s_tmax'] or float('inf')

        from .modules.shared import opts as opts_global
        from .modules.shared import opts_default

        for k,v in opts_default.__dict__.items():
            setattr(opts_global, k, v)

        opts = deepcopy(opts_default)
        [kwargs.pop(k, None) for k in [k for k in kwargs.keys() if 'info' in k or 'heading' in k or 'ㅤ' in k]]
        for k,v in kwargs.items():
            setattr(opts, k, v)

        first = first.clone()
        opts_key = 'smZ_opts'
        if type(first) is comfy.model_patcher.ModelPatcher:
            first.model_options.pop(opts_key, None)
            first.model_options[opts_key] = opts
            comfy.sample.prepare_noise = prepare_noise
            opts_global.debug = opts.debug
        elif type(first) is comfy.sd.CLIP:
            first.patcher.model_options.pop(opts_key, None)
            first.patcher.model_options[opts_key] = opts
            opts_global.debug = opts.debug
        return (first,)

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
