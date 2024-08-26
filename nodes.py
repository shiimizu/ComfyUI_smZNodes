import os
import logging
from pathlib import Path
from .modules import prompt_parser
from nodes import MAX_RESOLUTION
import comfy.model_patcher
import comfy.sd
import comfy.model_management
import comfy.samplers
from .modules.shared import logger
from .smZNodes import HijackClip, HijackClipComfy, convert_schedules_to_comfy

class smZ_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP", ),
                "parser": (["comfy", "comfy++", "A1111", "full", "compel", "fixed attention"],{"default": "comfy"}),
                "mean_normalization": ("BOOLEAN", {"default": True, "tooltip": "Toggles whether weights are normalized by taking the mean"}),
                "multi_conditioning": ("BOOLEAN", {"default": False}),
                "use_old_emphasis_implementation": ("BOOLEAN", {"default": False}),
                "with_SDXL": ("BOOLEAN", {"default": False}),
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
        if parser == 'comfy':
            with HijackClipComfy(clip) as clip:
                from nodes import CLIPTextEncode
                return CLIPTextEncode().encode(clip, text)

        if parser == 'comfy++':
            with HijackClipComfy(clip) as clip:
                tokens = clip.tokenize(text)
            with HijackClip(clip, mean_normalization) as clip:
                output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            return ([[cond, output]], )

        with HijackClip(clip, mean_normalization) as clip:
            model = lambda txt: clip.encode_from_tokens(clip.tokenize(txt), return_pooled=True, return_dict=True)
            schedules = prompt_parser.get_learned_conditioning(model, [text], smZ_steps, None, False)
            schedules_c = convert_schedules_to_comfy(schedules, with_steps=len(schedules[0])>1)

        from .modules.shared import opts, opts_default
        debug=opts.debug # get global opts' debug
        if (opts_new := clip.patcher.model_options.get('smZ_opts', None)) is not None:
            for k,v in opts_new.__dict__.items():
                setattr(opts, k, v)
            debug = opts_new.debug
        else:
            for k,v in opts_default.__dict__.items():
                setattr(opts, k, v)
        opts.debug = debug
        opts.prompt_mean_norm = mean_normalization
        opts.use_old_emphasis_implementation = use_old_emphasis_implementation
        opts.multi_conditioning = multi_conditioning
        class_name = clip.cond_stage_model.__class__.__name__ 
        is_sdxl = "SDXL" in class_name
        parsers = {
            "full": "Full parser",
            "compel": "Compel parser",
            "A1111": "A1111 parser",
            "fixed attention": "Fixed attention",
            "comfy++": "Comfy++ parser",
        }
        opts.prompt_attention = parsers.get(parser, "Comfy parser")
        if with_SDXL and is_sdxl:
            if class_name == "SDXLClipModel":
                for cc in schedules_c:
                    for cx in cc: # cc: [[]]
                        cx[1] |= { "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}
            elif class_name == "SDXLRefinerClipModel":
                for cc in schedules_c:
                    for cx in cc: # cc: [[]]
                        cx[1] |= {"aesthetic_score": ascore, "width": width,"height": height}
        return tuple(schedules_c)

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
        from .text_processing.emphasis import get_options_descriptions_nl
        i = 0
        def create_heading():
            nonlocal i
            return "ㅤ"*(i:=i+1)
        create_heading_value = lambda x: ("STRING", {"multiline": False, "default": x, "placeholder": x})
        optional = {
            # "show_headings": ("BOOLEAN", {"default": True}),
            # "show_descriptions": ("BOOLEAN", {"default":True}),

            create_heading(): create_heading_value("Stable Diffusion"),
            "info_comma_padding_backtrack": ("STRING", {"multiline": True, "placeholder": "Prompt word wrap length limit\nin tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"}),
            "Prompt word wrap length limit": ("INT", {"default": opts.comma_padding_backtrack, "min": 0, "max": 74, "step": 1, "tooltip": "🚧Prompt word wrap length limit\n\nin tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"}),
            "enable_emphasis": ("BOOLEAN", {"default": opts.enable_emphasis, "tooltip": "🚧Emphasis mode\n\nmakes it possible to make model to pay (more:1.1) or (less:0.9) attention to text when you use the syntax in prompt;\n\n" + get_options_descriptions_nl()}),

            "info_RNG": ("STRING", {"multiline": True, "placeholder": "Random number generator source.\nchanges seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"}),
            "RNG": (["cpu", "gpu", "nv"],{"default": opts.randn_source, "tooltip": "Random number generator source.\n\nchanges seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"}),
            
            create_heading(): create_heading_value("Compute Settings"),
            "info_disable_nan_check": ("STRING", {"multiline": True, "placeholder": "Disable NaN check in produced images/latent spaces. Only for CFGDenoiser."}),
            "disable_nan_check": ("BOOLEAN", {"default": opts.disable_nan_check, "tooltip": "Disable NaN check in produced images/latent spaces. Only for CFGDenoiser."}),

            create_heading(): create_heading_value("Sampler parameters"),
            "info_eta_ancestral": ("STRING", {"multiline": True, "placeholder": "Eta for k-diffusion samplers\nnoise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"}),
            "eta": ("FLOAT", {"default": opts.eta, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Eta for k-diffusion samplers\n\nnoise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"}),
            "info_s_churn": ("STRING", {"multiline": True, "placeholder": "Sigma churn\namount of stochasticity; only applies to Euler, Heun, Heun++2, and DPM2"}),
            "s_churn": ("FLOAT", {"default": opts.s_churn, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Sigma churn\n\namount of stochasticity; only applies to Euler, Heun, Heun++2, and DPM2"}),
            "info_s_tmin": ("STRING", {"multiline": True, "placeholder": "Sigma tmin\nenable stochasticity; start value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2'"}),
            "s_tmin": ("FLOAT", {"default": opts.s_tmin, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Sigma tmin\n\nenable stochasticity; start value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2'"}),
            "info_s_tmax": ("STRING", {"multiline": True, "placeholder": "Sigma tmax\n0 = inf; end value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2"}),
            "s_tmax": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "step": 0.01, "tooltip": "Sigma tmax\n\n0 = inf; end value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2"}),
            "info_s_noise": ("STRING", {"multiline": True, "placeholder": "Sigma noise\namount of additional noise to counteract loss of detail during sampling"}),
            "s_noise": ("FLOAT", {"default": opts.s_noise, "min": 0.0, "max": 1.1, "step": 0.001, "tooltip": "Sigma noise\n\namount of additional noise to counteract loss of detail during sampling"}),
            "info_eta_noise_seed_delta": ("STRING", {"multiline": True, "placeholder": "Eta noise seed delta\ndoes not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"}),
            "ENSD": ("INT", {"default": opts.eta_noise_seed_delta, "min": 0, "max": 0xffffffffffffffff, "step": 1, "tooltip": "Eta noise seed delta\n\ndoes not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"}),
            "info_skip_early_cond": ("STRING", {"multiline": True, "placeholder": "Ignore negative prompt during early sampling\ndisables CFG on a proportion of steps at the beginning of generation; 0=skip none; 1=skip all; can both improve sample diversity/quality and speed up sampling"}),
            "skip_early_cond": ("FLOAT", {"default": opts.skip_early_cond, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ignore negative prompt during early sampling\n\ndisables CFG on a proportion of steps at the beginning of generation; 0=skip none; 1=skip all; can both improve sample diversity/quality and speed up sampling"}),
            "info_sgm_noise_multiplier": ("STRING", {"multiline": True, "placeholder": "SGM noise multiplier\nmatch initial noise to official SDXL implementation - only useful for reproducing images\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818"}),
            "sgm_noise_multiplier": ("BOOLEAN", {"default": opts.sgm_noise_multiplier, "tooltip": "SGM noise multiplier\n\nmatch initial noise to official SDXL implementation - only useful for reproducing images\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818"}),
            "info_upcast_sampling": ("STRING", {"multiline": True, "placeholder": "upcast sampling.\nNo effect with --force-fp32. Usually produces similar results to --force-fp32 with better performance while using less memory."}),
            "upcast_sampling": ("BOOLEAN", {"default": opts.upcast_sampling, "tooltip": "🚧upcast sampling.\n\nNo effect with --force-fp32. Usually produces similar results to --force-fp32 with better performance while using less memory."}),

            create_heading(): create_heading_value("Optimizations"),
            "info_NGMS": ("STRING", {"multiline": True, "placeholder": "Negative Guidance minimum sigma\nskip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster. Only for CFGDenoiser.\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177\nhttps://github.com/lllyasviel/stable-diffusion-webui-forge/pull/1434"}),
            "NGMS": ("FLOAT", {"default": opts.s_min_uncond, "min": 0.0, "max": 15.0, "step": 0.01, "tooltip": "Negative Guidance minimum sigma\n\nskip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster.\nsee https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177\nhttps://github.com/lllyasviel/stable-diffusion-webui-forge/pull/1434"}),
            "info_NGMS_all_steps": ("STRING", {"multiline": True, "placeholder": "Negative Guidance minimum sigma all steps\nBy default, NGMS above skips every other step; this makes it skip all steps"}),
            "NGMS all steps": ("BOOLEAN", {"default": opts.s_min_uncond_all, "tooltip": "Negative Guidance minimum sigma all steps\n\nBy default, NGMS above skips every other step; this makes it skip all steps"}),
            "info_pad_cond_uncond": ("STRING", {"multiline": True, "placeholder": "Pad prompt/negative prompt to be same length\nimproves performance when prompt and negative prompt have different lengths; changes seeds. Only for CFGDenoiser."}),
            "pad_cond_uncond": ("BOOLEAN", {"default": opts.pad_cond_uncond, "tooltip": "🚧Pad prompt/negative prompt to be same length\n\nimproves performance when prompt and negative prompt have different lengths; changes seeds. Only for CFGDenoiser."}),
            "info_batch_cond_uncond": ("STRING", {"multiline": True, "placeholder": "Batch cond/uncond\ndo both conditional and unconditional denoising in one batch; uses a bit more VRAM during sampling, but improves speed. Only for CFGDenoiser."}),
            "batch_cond_uncond": ("BOOLEAN", {"default": opts.batch_cond_uncond, "tooltip": "🚧Batch cond/uncond\n\ndo both conditional and unconditional denoising in one batch; uses a bit more VRAM during sampling, but improves speed. Only for CFGDenoiser."}),

            create_heading(): create_heading_value("Compatibility"),
            "info_use_prev_scheduling": ("STRING", {"multiline": True, "placeholder": "Previous prompt editing timelines\nFor [red:green:N]; previous: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"}),
            "Use previous prompt editing timelines": ("BOOLEAN", {"default": opts.use_old_scheduling, "tooltip": "🚧Previous prompt editing timelines\n\nFor [red:green:N]; previous: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"}),
            
            create_heading(): create_heading_value("Experimental"),
            "info_use_CFGDenoiser": ("STRING", {"multiline": True, "placeholder": "CFGDenoiser\nAn experimental option to use stable-diffusion-webui's denoiser. It allows you to use the 'Optimizations' settings listed here."}),
            "Use CFGDenoiser": ("BOOLEAN", {"default": opts.use_CFGDenoiser, "tooltip": "🚧CFGDenoiser\n\nAn experimental option to use stable-diffusion-webui's denoiser. It allows you to use the 'Optimizations' settings listed here."}),
            "info_debug": ("STRING", {"multiline": True, "placeholder": "Debugging messages in the console."}),
            "debug": ("BOOLEAN", {"default": opts.debug, "label_on": "on", "label_off": "off", "tooltip": "Debugging messages in the console."}),
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
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",)

    def run(self, *args, **kwargs):
        first = kwargs.pop('*', None) if '*' in kwargs else args[0]
        if not hasattr(first, 'clone') or first is None: return (first,)

        kwargs['s_min_uncond'] = kwargs.pop('NGMS', 0.0)
        kwargs['s_min_uncond_all'] = kwargs.pop('NGMS all steps', False)
        kwargs['comma_padding_backtrack'] = kwargs.pop('Prompt word wrap length limit')
        kwargs['use_old_scheduling']=kwargs.pop("Use previous prompt editing timelines")
        kwargs['use_CFGDenoiser'] = kwargs.pop("Use CFGDenoiser")
        kwargs['randn_source'] = kwargs.pop('RNG')
        kwargs['eta_noise_seed_delta'] = kwargs.pop('ENSD')
        kwargs['s_tmax'] = kwargs['s_tmax'] or float('inf')

        from .modules.shared import opts_default, opts as opts_global

        for k,v in opts_default.__dict__.items():
            setattr(opts_global, k, v)

        opts = opts_default.clone()
        [kwargs.pop(k, None) for k in [k for k in kwargs.keys() if 'info' in k or 'heading' in k or 'ㅤ' in k]]
        for k,v in kwargs.items():
            setattr(opts, k, v)

        first = first.clone()
        opts_key = 'smZ_opts'
        if isinstance(first, comfy.model_patcher.ModelPatcher):
            first.model_options.pop(opts_key, None)
            first.model_options[opts_key] = opts
            opts_global.debug = opts.debug
        elif isinstance(first, comfy.sd.CLIP):
            first.patcher.model_options.pop(opts_key, None)
            first.patcher.model_options[opts_key] = opts
            opts_global.debug = opts.debug
        logger.setLevel(logging.DEBUG if opts_global.debug else logging.INFO)
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
