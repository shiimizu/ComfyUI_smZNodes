from .modules import prompt_parser, devices, sd_hijack_optimizations, shared
from .modules.shared import opts
from .modules.sd_hijack import model_hijack, list_optimizers
from .modules import sd_hijack
from .smZNodes import encode_from_tokens_with_custom_mean, encode_from_texts
from comfy.cli_args import args
from comfy.sdxl_clip import SDXLClipModel
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL
from nodes import MAX_RESOLUTION, CLIPTextEncode
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
                "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}), "clip": ("CLIP", ),
                "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}), "clip": ("CLIP", ),
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
        devices.device = clip.patcher.load_device
        shared.device = devices.device
        is_sdxl = type(clip.cond_stage_model) == SDXLClipModel

        dtype = torch.float16 if comfy.model_management.should_use_fp16(device=devices.device) else torch.float32
        devices.dtype_unet = torch.float16 if is_sdxl and not comfy.model_management.FORCE_FP32 else (_dtype if (_dtype:=clip.patcher.model_dtype() != None) else dtype)
        devices.unet_needs_upcast = shared.opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16
        devices.dtype = devices.dtype_unet
        devices.dtype_vae = comfy.model_management.vae_dtype()

        def run(steps=0, with_pooled=None):
            opts.prompt_mean_norm = mean_normalization
            opts.use_old_emphasis_implementation = use_old_emphasis_implementation
            opts.CLIP_stop_at_last_layers = abs(clip.layer_idx or 1)
            if is_sdxl:
                # Prevents tensor shape mismatch
                shared.cmd_opts.always_batch_cond_uncond = True
                shared.batch_cond_uncond = True
                
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

            sdxl_conds = {}
            if with_SDXL and is_sdxl:
                sdxl_conds = {
                    "aesthetic_score": ascore, "width": width, "height": height,
                    "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width,
                    "target_height": target_height, "text_g": text_g, "text_l": text_l
                }
            pooled={}
            if parser == "comfy":
                if with_SDXL and is_sdxl:
                    out = CLIPTextEncodeSDXL().encode(clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)
                    out[0][0][1]['aesthetic_score'] = sdxl_conds['aesthetic_score']
                    return out
                else:
                    return CLIPTextEncode().encode(clip, text)
            elif parser == "comfy++":
                tokens = clip.tokenize(text)

                def encode_toks(__tokens, clip_type="_clip_"):
                    clip_clone = clip.clone()
                    model_hijack.hijack(clip_clone)
                    try:
                        zs = []
                        first_pooled = None
                        for batch_chunk in __tokens:
                            tokens_ = [x[0] for x in batch_chunk]
                            multipliers = [x[1] for x in batch_chunk]
                            z = getattr(model_hijack.cond_stage_model, clip_type, model_hijack.cond_stage_model).process_tokens([tokens_], [multipliers])
                            if first_pooled == None:
                                first_pooled = z.pooled
                            zs.append(z)
                        zcond = torch.hstack(zs)
                        zcond.pooled = first_pooled
                        model_hijack.undo_hijack(clip_clone)
                    except Exception as err:
                        model_hijack.undo_hijack(clip_clone)
                        raise err
                    return zcond
                
                def encode_token_weights_custom(toks):
                    if is_sdxl and isinstance(toks, dict):
                        tok_g = toks['g']
                        tok_l = toks['l']
                        g_out = encode_toks(tok_g, "clip_g")
                        l_out = encode_toks(tok_l, "clip_l")
                        cond = torch.cat([l_out, g_out], dim=-1)
                        pooled = g_out.pooled
                    else:
                        cond = encode_toks(tokens)
                        pooled = cond.pooled
                    return (cond, pooled)
                
                clip_clone = clip.clone()
                clip_clone.cond_stage_model.encode_token_weights_orig = clip_clone.cond_stage_model.encode_token_weights
                clip_clone.cond_stage_model.encode_token_weights = encode_token_weights_custom
                cond, pooled = clip_clone.encode_from_tokens(tokens, return_pooled=True)
                clip_clone.cond_stage_model.encode_token_weights = clip_clone.cond_stage_model.encode_token_weights_orig
                # cond, pooled = encode_from_tokens_with_custom_mean(clip, tokens, return_pooled=True)
                # cond=zcond
                pooled = {"pooled_output": pooled, "from_smZ": True, "use_CFGDenoiser": use_CFGDenoiser, **sdxl_conds}
                return ([[cond, pooled ]], )
            else:
                texts = [text]
                # Because of prompt editing, we need the total number of steps
                # So this function will be called back at the sampling stage
                # if steps == 0:
                #     texts = ['']
                #     steps = 1
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
                    _cond, pooled = encode_from_texts(clip_clone, texts, steps=steps, return_pooled=True, multi=multi_conditioning, with_pooled=with_pooled)
                    _cond.cond = pooled.cond
                    cond = cond if cond != None else _cond
                    model_hijack.undo_hijack(clip_clone)
                except Exception as err:
                    model_hijack.undo_hijack(clip_clone)
                    raise err
                pooled = {"pooled_output": pooled, "from_smZ": True, "use_CFGDenoiser": use_CFGDenoiser, "schedules_": cond.cond, **sdxl_conds}
            return ([[cond, pooled if with_pooled == None else with_pooled]], )

        result = run()
        result[0][0][1]['encode_fn'] = run
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