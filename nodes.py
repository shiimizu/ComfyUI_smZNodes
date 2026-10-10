import re
import logging
from itertools import chain
from nodes import MAX_RESOLUTION
import comfy.model_patcher
import comfy.sd
import comfy.model_management
import comfy.samplers
from .smZNodes import HijackClip, HijackClipComfy, get_learned_conditioning
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL

# Detect ComfyUI V3 API
try:
    from comfy_api.latest import io
    V3_API = True
except ImportError:
    try:
        from comfy_api.v0_0_2 import ComfyExtension, io
        V3_API = True
    except ImportError:
        io = None
        V3_API = False

# from comfy_api.latest import io as io
# from comfy_api.latest import ComfyExtension, io

# BaseComfyNode = io.ComfyNode
BaseComfyNode = io.ComfyNode if V3_API else object


class smZ_CLIPTextEncode(BaseComfyNode):
    PARSER_MAP = {
        "full": "Full parser",
        "compel": "Compel parser",
        "A1111": "A1111 parser",
        "fixed attention": "Fixed attention",
        "comfy++": "Comfy++ parser",
    }

    @classmethod
    def define_schema(cls) -> "io.Schema":
        # Shared parser flags defined once at the node root
        shared_parser_inputs = [
            io.Combo.Input("parser", options=["comfy", "comfy++", "A1111", "full", "compel", "fixed attention"], default="comfy"),
            io.Boolean.Input("mean_normalization", default=True, tooltip="Mean normalize attention weights"),
            io.Boolean.Input("multi_conditioning", default=True, tooltip="Split prompt by 'AND'"),
            io.Boolean.Input("use_old_emphasis_implementation", default=False, advanced=True, tooltip="Legacy A1111 emphasis"),
        ]
    
        # Only mode-specific fields stay in DynamicCombo
        mode_options = [
            io.DynamicCombo.Option("Standard", [
                io.String.Input("text", multiline=True, dynamic_prompts=True, default="", placeholder="text"),
            ]),
            io.DynamicCombo.Option("SDXL", [
                io.String.Input("text_g", multiline=True, dynamic_prompts=True, default="", placeholder="text_g"),
                io.String.Input("text_l", multiline=True, dynamic_prompts=True, default="", placeholder="text_l"),
                io.Float.Input("ascore", default=6.0, min=0.0, max=1000.0, step=0.01),
                io.Int.Input("width", default=1024, min=0, max=MAX_RESOLUTION),
                io.Int.Input("height", default=1024, min=0, max=MAX_RESOLUTION),
                io.Int.Input("crop_w", default=0, min=0, max=MAX_RESOLUTION, advanced=True),
                io.Int.Input("crop_h", default=0, min=0, max=MAX_RESOLUTION, advanced=True),
                io.Int.Input("target_width", default=1024, min=0, max=MAX_RESOLUTION, advanced=True),
                io.Int.Input("target_height", default=1024, min=0, max=MAX_RESOLUTION, advanced=True),
            ]),
        ]
    
        return io.Schema(
            node_id="smZ CLIPTextEncode",
            display_name="CLIP Text Encode++",
            category="conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.DynamicCombo.Input("mode", options=mode_options, tooltip="Encoding mode"),
                *shared_parser_inputs,
                io.Int.Input("smZ_steps", default=1, min=1, max=0xffffffffffffffff, optional=True, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="CONDITIONING")
            ],
        )

    # Legacy V1 INPUT_TYPES fallback
    # @classmethod
    # def INPUT_TYPES(s):
    #     return {
    #         "required": {
    #             "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
    #             "clip": ("CLIP", ),
    #             "parser": (["comfy", "comfy++", "A1111", "full", "compel", "fixed attention"], {"default": "comfy"}),
    #             "mean_normalization": ("BOOLEAN", {"default": True, "tooltip": "Toggles whether weights are normalized by taking the mean"}),
    #             "multi_conditioning": ("BOOLEAN", {"default": True}),
    #             "use_old_emphasis_implementation": ("BOOLEAN", {"default": False}),
    #             "with_SDXL": ("BOOLEAN", {"default": False}),
    #             "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
    #             "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
    #             "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
    #             "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
    #             "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
    #             "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
    #             "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
    #             "text_g": ("STRING", {"multiline": True, "placeholder": "CLIP_G", "dynamicPrompts": True}),
    #             "text_l": ("STRING", {"multiline": True, "placeholder": "CLIP_L", "dynamicPrompts": True}),
    #         },
    #         "optional": {
    #             "smZ_steps": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
    #         },
    #     }

    # RETURN_TYPES = ("CONDITIONING",)
    # FUNCTION = "execute"
    # CATEGORY = "conditioning"

    @classmethod
    def execute(cls, clip: comfy.sd.CLIP, parser="comfy", mean_normalization=True,
                use_old_emphasis_implementation=False, mode=None, smZ_steps=1, **kwargs):
        # Resolve inputs whether coming from DynamicCombo or legacy kwargs
        if isinstance(mode, dict):
            selected = mode.get("mode", "Standard").lower()
            if selected == "sdxl":
                with_SDXL = True
                text = kwargs.get("text", "")
                multi_conditioning = False
                text_g = mode.get("text_g", "")
                text_l = mode.get("text_l", "")
                ascore = float(mode.get("ascore", 6.0))
                width = int(mode.get("width", 1024))
                height = int(mode.get("height", 1024))
                crop_w = int(mode.get("crop_w", 0))
                crop_h = int(mode.get("crop_h", 0))
                target_width = int(mode.get("target_width", 1024))
                target_height = int(mode.get("target_height", 1024))
            else:
                with_SDXL = False
                text = mode.get("text", "")
                multi_conditioning = mode.get("multi_conditioning", True)
                text_g = ""
                text_l = ""
                ascore = 6.0
                width = 1024
                height = 1024
                crop_w = 0
                crop_h = 0
                target_width = 1024
                target_height = 1024
        else:
            with_SDXL = kwargs.get("with_SDXL", False)
            text = kwargs.get("text", "")
            multi_conditioning = kwargs.get("multi_conditioning", True)
            text_g = kwargs.get("text_g", "")
            text_l = kwargs.get("text_l", "")
            ascore = float(kwargs.get("ascore", 6.0))
            width = int(kwargs.get("width", 1024))
            height = int(kwargs.get("height", 1024))
            crop_w = int(kwargs.get("crop_w", 0))
            crop_h = int(kwargs.get("crop_h", 0))
            target_width = int(kwargs.get("target_width", 1024))
            target_height = int(kwargs.get("target_height", 1024))

        from .modules.shared import Options, opts, opts_default
        debug = opts.debug
        if (opts_new := clip.patcher.model_options.get(Options.KEY, None)) is not None:
            opts.update(opts_new)
            debug = opts_new.debug
        else:
            opts.update(opts_default)
        opts.debug = debug
        opts.prompt_mean_norm = mean_normalization
        opts.use_old_emphasis_implementation = use_old_emphasis_implementation
        opts.multi_conditioning = multi_conditioning

        class_name = clip.cond_stage_model.__class__.__name__
        is_sdxl = "SDXL" in class_name
        on_sdxl = with_SDXL and is_sdxl
        opts.prompt_attention = cls.PARSER_MAP.get(parser, "Comfy parser")

        def _comfy_path(c_clip, c_text):
            if on_sdxl and class_name == "SDXLClipModel":
                return CLIPTextEncodeSDXL().encode(c_clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)
            elif on_sdxl and class_name == "SDXLRefinerClipModel":
                from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXLRefiner
                return CLIPTextEncodeSDXLRefiner().encode(c_clip, c_clip, ascore, width, height, c_text)
            else:
                from nodes import CLIPTextEncode
                return CLIPTextEncode().encode(c_clip, c_text)

        def comfy_path(c_clip):
            if on_sdxl and class_name == "SDXLRefinerClipModel":
                return _comfy_path(c_clip, text)
            else:
                if multi_conditioning:
                    prompts = re.compile(r"\bAND\b").split(text)
                    return (list(chain(*(_comfy_path(c_clip, prompt)[0] for prompt in prompts))), )
                else:
                    return _comfy_path(c_clip, text)

        if parser == "comfy":
            with HijackClipComfy(clip) as h_clip:
                schedules = comfy_path(h_clip)[0]
        elif parser == "comfy++":
            with HijackClip(clip, opts) as h_clip:
                with HijackClipComfy(h_clip) as h_clip_c:
                    schedules = comfy_path(h_clip_c)[0]
        else:
            with HijackClip(clip, opts) as h_clip:
                model = lambda txt: h_clip.encode_from_tokens(h_clip.tokenize(txt), return_pooled=True, return_dict=True)
                steps = max(smZ_steps, 1)
                if on_sdxl and class_name == "SDXLClipModel":
                    schedules = CLIPTextEncodeSDXL().encode(h_clip, width, height, crop_w, crop_h, target_width, target_height, [text_g], [text_l])[0]
                else:
                    schedules = get_learned_conditioning(model, [text], steps, multi_conditioning)

        if on_sdxl and class_name == "SDXLRefinerClipModel":
            for cx in schedules:
                cx[1] |= {"aesthetic_score": ascore, "width": width, "height": height}

        if V3_API:
            return io.NodeOutput(schedules)
        return (schedules,)

    encode = execute

class smZ_Settings(BaseComfyNode):
    @classmethod
    def define_schema(cls) -> "io.Schema":
        from .modules.shared import opts_default
        template = io.MatchType.Template("*")
        # Helper to generate unique Hangul filler (\u3164) inputs for headings
        h_idx = 0
        def create_heading(title: str):
            nonlocal h_idx
            h_idx += 1
            return io.String.Input("ㅤ" * h_idx, default=title, socketless=True)

        return io.Schema(
            node_id="smZ Settings",
            display_name="Settings (smZ)",
            category="advanced",
            search_aliases=["smZ Settings", "Settings (smZ)"],
            accept_all_inputs=True,
            inputs=[
                io.MatchType.Input("*", template=template, tooltip="Model or CLIP to attach smZ settings to"),

                # --- Stable Diffusion ---
                create_heading("Stable Diffusion"),
                io.Int.Input("prompt_word_wrap_length_limit", default=opts_default.comma_padding_backtrack, min=0, max=255, step=1, display_name="Prompt word wrap length limit", tooltip="Prompt word wrap length limit in tokens - moves short words to next chunk if exceeding 75 tokens"),
                io.Boolean.Input("enable_emphasis", default=opts_default.enable_emphasis, tooltip="Enable emphasis syntax (e.g. (text:1.1))"),
                io.Combo.Input("RNG", options=["cpu", "gpu", "nv"], default=opts_default.randn_source, tooltip="Random number generator source"),

                # --- Compute Settings ---
                create_heading("Compute Settings"),
                io.Boolean.Input("disable_nan_check", default=opts_default.disable_nan_check, tooltip="Disable NaN check in produced latents/images (CFGDenoiser only)"),

                # --- Sampler parameters ---
                create_heading("Sampler parameters"),
                io.Float.Input("eta", default=opts_default.eta, min=0.0, max=1.0, step=0.01, tooltip="Eta for k-diffusion ancestral and SDE samplers"),
                io.Float.Input("s_churn", default=opts_default.s_churn, min=0.0, max=100.0, step=0.01, tooltip="Sigma churn: amount of stochasticity"),
                io.Float.Input("s_tmin", default=opts_default.s_tmin, min=0.0, max=10.0, step=0.01, tooltip="Sigma tmin: start value for stochasticity"),
                io.Float.Input("s_tmax", default=0.0, min=0.0, max=999.0, step=0.01, tooltip="Sigma tmax: end value for stochasticity (0 = inf)"),
                io.Float.Input("s_noise", default=opts_default.s_noise, min=0.0, max=1.1, step=0.001, tooltip="Sigma noise: additional noise to counteract loss of detail"),
                io.Int.Input("ENSD", default=opts_default.eta_noise_seed_delta, min=0, max=0xffffffffffffffff, step=1, tooltip="Eta noise seed delta for reproducing ancestral sampler outputs"),
                io.Float.Input("skip_early_cond", default=opts_default.skip_early_cond, min=0.0, max=1.0, step=0.01, tooltip="Fraction of steps at the beginning to skip negative prompt guidance"),
                io.Boolean.Input("sgm_noise_multiplier", default=opts_default.sgm_noise_multiplier, tooltip="Match initial noise to official SDXL implementation"),
                io.Boolean.Input("upcast_sampling", default=opts_default.upcast_sampling, tooltip="🚧 Upcast sampling precision to FP32"),

                # --- Optimizations ---
                create_heading("Optimizations"),
                io.Float.Input("NGMS", default=opts_default.s_min_uncond, min=0.0, max=15.0, step=0.01, tooltip="Negative Guidance minimum sigma: skip negative prompt at low sigmas"),
                io.Boolean.Input("NGMS_all_steps", default=opts_default.s_min_uncond_all, display_name="NGMS all steps", tooltip="🚧 Skip negative prompt on all steps below NGMS"),
                io.Boolean.Input("pad_cond_uncond", default=opts_default.pad_cond_uncond, tooltip="🚧 Pad prompt and negative prompt to equal length for CFGDenoiser"),
                io.Boolean.Input("batch_cond_uncond", default=opts_default.batch_cond_uncond, tooltip="🚧 Batch conditional and unconditional denoising (CFGDenoiser)"),

                # --- Compatibility ---
                create_heading("Compatibility"),
                io.Boolean.Input("use_old_scheduling", default=opts_default.use_old_scheduling, display_name="Use previous prompt editing timelines", tooltip="🚧 Compatibility with legacy prompt editing step scheduling"),

                # --- Experimental ---
                create_heading("Experimental"),
                io.Boolean.Input("use_CFGDenoiser", default=opts_default.use_CFGDenoiser, display_name="Use CFGDenoiser", tooltip="🚧 Use stable-diffusion-webui's denoiser pipeline"),
                io.Boolean.Input("debug", default=opts_default.debug, tooltip="Print debug messages in console"),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="*", tooltip="Model or CLIP with smZ options attached")
            ]
        )

    # Legacy V1 INPUT_TYPES fallback
    # @classmethod
    # def INPUT_TYPES(s):
    #     from .modules.shared import opts_default as opts
    #     i = 0
    #     def create_heading():
    #         nonlocal i
    #         return "ㅤ" * (i := i + 1)
    #     create_heading_value = lambda x: ("STRING", {"multiline": False, "default": x, "placeholder": x})
        
    #     return {
    #         "required": {
    #             "*": ("*", {"forceInput": True}),
    #         },
    #         "optional": {
    #             create_heading(): create_heading_value("Stable Diffusion"),
    #             "info_comma_padding_backtrack": ("STRING", {"multiline": True, "placeholder": "Prompt word wrap length limit\nin tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"}),
    #             "Prompt word wrap length limit": ("INT", {"default": opts.comma_padding_backtrack, "min": 0, "max": 255, "step": 1}),
    #             "enable_emphasis": ("BOOLEAN", {"default": opts.enable_emphasis}),

    #             "info_RNG": ("STRING", {"multiline": True, "placeholder": "Random number generator source.\nchanges seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"}),
    #             "RNG": (["cpu", "gpu", "nv"], {"default": opts.randn_source}),

    #             create_heading(): create_heading_value("Compute Settings"),
    #             "info_disable_nan_check": ("STRING", {"multiline": True, "placeholder": "Disable NaN check in produced images/latent spaces. Only for CFGDenoiser."}),
    #             "disable_nan_check": ("BOOLEAN", {"default": opts.disable_nan_check}),

    #             create_heading(): create_heading_value("Sampler parameters"),
    #             "info_eta_ancestral": ("STRING", {"multiline": True, "placeholder": "Eta for k-diffusion samplers\nnoise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"}),
    #             "eta": ("FLOAT", {"default": opts.eta, "min": 0.0, "max": 1.0, "step": 0.01}),
    #             "info_s_churn": ("STRING", {"multiline": True, "placeholder": "Sigma churn\namount of stochasticity; only applies to Euler, Heun, Heun++2, and DPM2"}),
    #             "s_churn": ("FLOAT", {"default": opts.s_churn, "min": 0.0, "max": 100.0, "step": 0.01}),
    #             "info_s_tmin": ("STRING", {"multiline": True, "placeholder": "Sigma tmin\nenable stochasticity; start value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2"}),
    #             "s_tmin": ("FLOAT", {"default": opts.s_tmin, "min": 0.0, "max": 10.0, "step": 0.01}),
    #             "info_s_tmax": ("STRING", {"multiline": True, "placeholder": "Sigma tmax\n0 = inf; end value of the sigma range; only applies to Euler, Heun, Heun++2, and DPM2"}),
    #             "s_tmax": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "step": 0.01}),
    #             "info_s_noise": ("STRING", {"multiline": True, "placeholder": "Sigma noise\namount of additional noise to counteract loss of detail during sampling"}),
    #             "s_noise": ("FLOAT", {"default": opts.s_noise, "min": 0.0, "max": 1.1, "step": 0.001}),
    #             "info_eta_noise_seed_delta": ("STRING", {"multiline": True, "placeholder": "Eta noise seed delta\ndoes not improve anything, just produces different results for ancestral samplers"}),
    #             "ENSD": ("INT", {"default": opts.eta_noise_seed_delta, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
    #             "info_skip_early_cond": ("STRING", {"multiline": True, "placeholder": "Ignore negative prompt during early sampling\ndisables CFG on a proportion of steps at the beginning of generation"}),
    #             "skip_early_cond": ("FLOAT", {"default": opts.skip_early_cond, "min": 0.0, "max": 1.0, "step": 0.01}),
    #             "info_sgm_noise_multiplier": ("STRING", {"multiline": True, "placeholder": "SGM noise multiplier\nmatch initial noise to official SDXL implementation"}),
    #             "sgm_noise_multiplier": ("BOOLEAN", {"default": opts.sgm_noise_multiplier}),
    #             "info_upcast_sampling": ("STRING", {"multiline": True, "placeholder": "upcast sampling.\nUsually produces similar results to --force-fp32 with better performance"}),
    #             "upcast_sampling": ("BOOLEAN", {"default": opts.upcast_sampling}),

    #             create_heading(): create_heading_value("Optimizations"),
    #             "info_NGMS": ("STRING", {"multiline": True, "placeholder": "Negative Guidance minimum sigma\nskip negative prompt for some steps when the image is almost ready; 0=disable"}),
    #             "NGMS": ("FLOAT", {"default": opts.s_min_uncond, "min": 0.0, "max": 15.0, "step": 0.01}),
    #             "info_NGMS_all_steps": ("STRING", {"multiline": True, "placeholder": "Negative Guidance minimum sigma all steps\nBy default, NGMS above skips every other step; this makes it skip all steps"}),
    #             "NGMS all steps": ("BOOLEAN", {"default": opts.s_min_uncond_all}),
    #             "info_pad_cond_uncond": ("STRING", {"multiline": True, "placeholder": "Pad prompt/negative prompt to be same length\nimproves performance when lengths differ; changes seeds"}),
    #             "pad_cond_uncond": ("BOOLEAN", {"default": opts.pad_cond_uncond}),
    #             "info_batch_cond_uncond": ("STRING", {"multiline": True, "placeholder": "Batch cond/uncond\ndo both conditional and unconditional denoising in one batch"}),
    #             "batch_cond_uncond": ("BOOLEAN", {"default": opts.batch_cond_uncond}),

    #             create_heading(): create_heading_value("Compatibility"),
    #             "info_use_prev_scheduling": ("STRING", {"multiline": True, "placeholder": "Previous prompt editing timelines\nFor [red:green:N]"}),
    #             "Use previous prompt editing timelines": ("BOOLEAN", {"default": opts.use_old_scheduling}),

    #             create_heading(): create_heading_value("Experimental"),
    #             "info_use_CFGDenoiser": ("STRING", {"multiline": True, "placeholder": "CFGDenoiser\nUse stable-diffusion-webui's denoiser"}),
    #             "Use CFGDenoiser": ("BOOLEAN", {"default": opts.use_CFGDenoiser}),
    #             "info_debug": ("STRING", {"multiline": True, "placeholder": "Debugging messages in the console."}),
    #             "debug": ("BOOLEAN", {"default": opts.debug}),
    #         }
    #     }

    # RETURN_TYPES = ("*",)
    # FUNCTION = "execute"
    # CATEGORY = "advanced"
    # OUTPUT_TOOLTIPS = ("The target (Model or CLIP) with smZ configuration attached.",)

    @classmethod
    def execute(cls, *args, **kwargs):
        first = kwargs.pop("*", None)
        if first is None and args:
            first = args[0]

        if first is None or not hasattr(first, 'clone'):
            return io.NodeOutput(first) if V3_API else (first,)

        # Normalize widget names from either legacy or modern schemas
        s_min_uncond = kwargs.pop('NGMS', kwargs.pop('s_min_uncond', 0.0))
        s_min_uncond_all = kwargs.pop('NGMS all steps', kwargs.pop('NGMS_all_steps', kwargs.pop('s_min_uncond_all', False)))
        comma_padding_backtrack = kwargs.pop('Prompt word wrap length limit', kwargs.pop('prompt_word_wrap_length_limit', kwargs.pop('comma_padding_backtrack', 20)))
        use_old_scheduling = kwargs.pop("Use previous prompt editing timelines", kwargs.pop("use_old_scheduling", False))
        use_CFGDenoiser = kwargs.pop("Use CFGDenoiser", kwargs.pop("use_CFGDenoiser", False))
        randn_source = kwargs.pop('RNG', kwargs.pop('randn_source', 'gpu'))
        eta_noise_seed_delta = kwargs.pop('ENSD', kwargs.pop('eta_noise_seed_delta', 0))
        s_tmax = kwargs.pop('s_tmax', 0.0) or float('inf')

        kwargs['s_min_uncond'] = s_min_uncond
        kwargs['s_min_uncond_all'] = s_min_uncond_all
        kwargs['comma_padding_backtrack'] = comma_padding_backtrack
        kwargs['use_old_scheduling'] = use_old_scheduling
        kwargs['use_CFGDenoiser'] = use_CFGDenoiser
        kwargs['randn_source'] = randn_source
        kwargs['eta_noise_seed_delta'] = eta_noise_seed_delta
        kwargs['s_tmax'] = s_tmax

        from .modules.shared import Options, logger, opts_default, opts as opts_global

        opts_global.update(opts_default)
        opts = opts_default.clone()
        kwargs_clean = {k: v for k, v in kwargs.items() if not ('info' in k or 'heading' in k or 'ㅤ' in k or k == 'extra')}
        opts.update(kwargs_clean)
        opts_global.debug = opts.debug

        opts_key = Options.KEY
        if isinstance(first, comfy.model_patcher.ModelPatcher):
            first = first.clone()
            first.model_options[opts_key] = opts
        elif isinstance(first, comfy.sd.CLIP):
            first = first.clone()
            first.patcher.model_options[opts_key] = opts

        logger.setLevel(logging.DEBUG if opts_global.debug else logging.INFO)
        if V3_API:
            return io.NodeOutput(first)
        return (first,)

    apply = execute


NODE_CLASS_MAPPINGS = {
    "smZ CLIPTextEncode": smZ_CLIPTextEncode,
    "smZ Settings": smZ_Settings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "smZ CLIPTextEncode": "CLIP Text Encode++",
    "smZ Settings": "Settings (smZ)",
}