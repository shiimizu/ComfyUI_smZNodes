from .modules import prompt_parser, devices, sd_hijack_optimizations, shared
from .modules.shared import opts
from .modules.sd_hijack import model_hijack, list_optimizers
from .modules import sd_hijack
from .smZNodes import encode_from_tokens_with_custom_mean, encode_from_texts
from comfy.cli_args import args
from comfy.sdxl_clip import SDXLClipModel
import comfy.sd
import comfy.model_management
import torch
import comfy.samplers
import comfy.sample

class smZ_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP", ),
                "parser": (["comfy", "comfy++", "A1111", "full", "compel", "fixed attention"],{"default": "comfy"}),
                # whether weights are normalized by taking the mean
                "mean_normalization": ([False, True],{"default": True}),
                "multi_conditioning": ([False, True],{"default": True}),
                "use_old_emphasis_implementation": ([False, True],{"default": False}),
                "use_CFGDenoiser": ([False, True],{"default": False}),
            },
            "hidden": {
                "with_SDXL": ([False, True],{"default": False}),
                "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}),
                "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip: comfy.sd.CLIP, text: str, parser: str, mean_normalization: bool, multi_conditioning: bool, use_old_emphasis_implementation: bool, use_CFGDenoiser:bool,with_SDXL=False,text_g="",text_l=""):
        devices.device = clip.patcher.load_device
        shared.device = devices.device
        # devices.device = comfy.model_management.get_torch_device()
        opts.data['prompt_mean_norm'] = mean_normalization
        opts.data['use_old_emphasis_implementation'] = use_old_emphasis_implementation
        opts.data['CLIP_stop_at_last_layers'] = abs(clip.layer_idx or 1)
        is_sdxl = type(clip.cond_stage_model) == SDXLClipModel
        if is_sdxl:
            # Prevents tensor shape mismatch
            shared.cmd_opts.always_batch_cond_uncond = True
            shared.batch_cond_uncond = True

        dtype = torch.float16 if comfy.model_management.should_use_fp16(device=devices.device) else torch.float32
        devices.dtype_unet = torch.float16 if is_sdxl and not comfy.model_management.FORCE_FP32 else (_dtype if (_dtype:=clip.patcher.model_dtype() != None) else dtype)
        devices.unet_needs_upcast = shared.opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16
        devices.dtype = devices.dtype_unet
        devices.dtype_vae = comfy.model_management.vae_dtype()

        def run():
            parser_d = {"full": "Full parser",
                 "compel": "Compel parser",
                 "A1111": "A1111 parser",
                 "fixed attention": "Fixed attention",
                 "comfy++": "Comfy++ parser",
                 }
            opts.prompt_attention = parser_d.get(parser, "Comfy parser")

            if parser != "comfy":
                opts.disable_max_denoise = True
                opts.use_CFGDenoiser = use_CFGDenoiser

            pooled={}
            if "comfy" in parser:
                tokens = clip.tokenize(text)
                if parser == "comfy++":
                    if is_sdxl:
                        raise NotImplementedError
                    clip_clone = clip.clone()
                    model_hijack.hijack(clip_clone)
                    try:
                        zs = []
                        for batch_chunk in tokens:
                            tokens_ = [x[0] for x in batch_chunk]
                            multipliers = [x[1] for x in batch_chunk]
                            z = model_hijack.cond_stage_model.process_tokens([tokens_], [multipliers])
                            zs.append(z)
                        zcond = torch.hstack(zs)
                        model_hijack.undo_hijack(clip_clone)
                    except Exception as err:
                        model_hijack.undo_hijack(clip_clone)
                        raise err

                    cond, pooled = encode_from_tokens_with_custom_mean(clip, tokens, return_pooled=True)
                    cond=zcond
                    pooled = {"pooled_output": pooled, "from_smZ": True, "use_CFGDenoiser": use_CFGDenoiser}
                else:
                    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                    pooled = {"pooled_output": pooled}
                return ([[cond, pooled ]], )
            else:
                texts = [text]
                create_prompts = lambda txts: prompt_parser.SdConditioning(txts)
                texts = create_prompts(texts)
                if type(clip.cond_stage_model).__name__ == "SDXLClipModel":
                    if with_SDXL:
                        texts = {"g": create_prompts([text_g]), "l": create_prompts([text_l])}
                    else:
                        texts = {"g": texts, "l": texts}

                clip_clone = clip.clone()
                model_hijack.hijack(clip_clone)
                try:
                    steps = 1
                    cond = None
                    # from A1111's processing.py and sd_samplers_kdiffusion.py
                    # if not is_sdxl:
                    #     if multi_conditioning:
                    #         c = prompt_parser.get_multicond_learned_conditioning(clip_clone.cond_stage_model, texts, steps)
                    #         conds_list, cond = prompt_parser.reconstruct_multicond_batch(c, steps)
                    #         cond.cond = c
                    #     else:
                    #         uc = prompt_parser.get_learned_conditioning(clip_clone.cond_stage_model, texts, steps)
                    #         cond = prompt_parser.reconstruct_cond_batch(uc, steps)
                    #         cond.cond = uc
                    _cond, pooled = encode_from_texts(clip_clone, texts, steps=steps, return_pooled=True, multi=multi_conditioning)
                    _cond.cond = pooled.cond
                    cond = cond if cond != None else _cond
                    model_hijack.undo_hijack(clip_clone)
                except Exception as err:
                    model_hijack.undo_hijack(clip_clone)
                    raise err
                sdxl_conds = {}
                pooled = {"pooled_output": pooled, "from_smZ": True, "use_CFGDenoiser": use_CFGDenoiser, "cond_":cond.cond, **sdxl_conds}
            return ([[cond, pooled ]], )

        result = run()
        return result

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "smZ CLIPTextEncode": smZ_CLIPTextEncode,
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "smZ CLIPTextEncode" : "CLIP Text Encode++",
}