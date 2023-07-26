from .modules import prompt_parser, devices
from .modules.shared import opts
from .modules.sd_hijack import model_hijack
from .smZNodes import encode_from_tokens_with_custom_mean, encode_from_texts, scaled_dot_product_no_mem_attention_forward, sdp_no_mem_attnblock_forward
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
        # devices.device = comfy.model_management.get_torch_device()
        opts.data['prompt_mean_norm'] = mean_normalization
        opts.data['use_old_emphasis_implementation'] = use_old_emphasis_implementation
        # not necessary since we use a different transform function
        opts.data['clip_skip'] = abs(clip.layer_idx or 1)

        def run():
            if parser == "full":
                opts.data['prompt_attention'] = "Full parser"
            elif parser == "compel":
                opts.data['prompt_attention'] = "Compel parser"
            elif parser == "A1111":
                opts.data['prompt_attention'] = "A1111 parser"
            elif parser == "fixed attention":
                opts.data['prompt_attention'] = "Fixed attention"
            elif parser == "comfy++":
                opts.data['prompt_attention'] = "Comfy++ parser"
            else:
                opts.data['prompt_attention'] = "Comfy parser"

            if parser != "comfy":
                opts.data['disable_max_denoise'] = True
                opts.data['use_CFGDenoiser'] = use_CFGDenoiser
                if comfy.model_management.get_torch_device() == torch.device("mps"):
                    try:
                        from torch import mps
                    except:
                        pass

            pooled=None
            if "comfy" in parser:
                tokens = clip.tokenize(text)
                if parser == "comfy++":

                    clip_clone = clip.clone()
                    model_hijack.hijack(clip_clone)
                    zs = []
                    for batch_chunk in tokens:
                        tokens_ = [x[0] for x in batch_chunk]
                        multipliers = [x[1] for x in batch_chunk]
                        z = model_hijack.cond_stage_model.process_tokens([tokens_], [multipliers])
                        zs.append(z)
                    zcond = torch.hstack(zs)
                    model_hijack.undo_hijack(clip_clone)

                    cond, pooled = encode_from_tokens_with_custom_mean(clip, tokens, return_pooled=True)
                    cond=zcond
                    pooled = {"pooled_output": pooled, "from_smZ": True, "use_CFGDenoiser": use_CFGDenoiser}
                else:
                    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                    pooled = {"pooled_output": pooled}
                return ([[cond, pooled or {} ]], )
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
                steps = 1
                # from A1111's processing.py and sd_samplers_kdiffusion.py
                if multi_conditioning:
                    # c = prompt_parser.get_multicond_learned_conditioning(clip_clone.cond_stage_model, texts, steps)
                    # conds_list, cond = prompt_parser.reconstruct_multicond_batch(c, steps)

                    _cond, pooled = encode_from_texts(clip_clone, texts, return_pooled=True, multi=True)
                else:
                    # uc = prompt_parser.get_learned_conditioning(clip_clone.cond_stage_model, texts, steps)
                    # cond = prompt_parser.reconstruct_cond_batch(uc, steps)

                    _cond, pooled = encode_from_texts(clip_clone, texts, return_pooled=True)
                model_hijack.undo_hijack(clip_clone)
                # if opts.use_old_emphasis_implementation:
                cond = _cond
                pooled = {"pooled_output": pooled, "from_smZ": True, "use_CFGDenoiser": use_CFGDenoiser}
                # print("cond (+)" if multi_conditioning else "uncond (-)", cond) # debug
            return ([[cond, pooled or {} ]], )

        result = run()
        # print("cond (+)" if multi_conditioning else "uncond (-)", result[0][0][0]) # debug
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