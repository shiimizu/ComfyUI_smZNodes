from .modules import prompt_parser
from .modules.shared import opts
from .modules.sd_hijack import model_hijack
from .smZNodes import encode_from_tokens_with_mean

class smZ_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "clip": ("CLIP", ),
            "parser": (["comfy", "comfy++", "A1111", "full", "compel", "fixed attention"],{"default": "comfy"}),
            # whether weights are normalized by taking the mean
            "mean_normalization": ([False, True],{"default": False}),
            "multi_conditioning": ([False, True],{"default": False}),
            },}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, parser, mean_normalization, multi_conditioning, prompt=None):
        opts.data['prompt_mean_norm'] = mean_normalization

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
            
            if "comfy" in parser:
                pooled=None
                tokenized = clip.tokenize(text, return_word_ids=mean_normalization)
                if parser == "comfy++":
                    cond, pooled = encode_from_tokens_with_mean(clip, tokenized, return_pooled=True)
                else:
                    cond, pooled = clip.encode_from_tokens(tokenized, return_pooled=True)
                return ([[cond, {} if pooled is None else {"pooled_output": pooled} ]], )
            else:
                # not necessary since we use a different transform function
                opts.data['clip_skip'] = abs(clip.layer_idx)
                
                texts = [text]
                model_hijack.hijack(clip)
                try:
                    # from A1111's processing.py and sd_samplers_kdiffusion.py
                    if multi_conditioning:
                        c = prompt_parser.get_multicond_learned_conditioning(clip.cond_stage_model, texts, 1)
                        conds_list, cond = prompt_parser.reconstruct_multicond_batch(c, 1)
                    else:
                        uc = prompt_parser.get_learned_conditioning(clip.cond_stage_model, texts, 1)
                        cond = prompt_parser.reconstruct_cond_batch(uc, 1)
                    model_hijack.undo_hijack(clip)
                except Exception as error:
                    model_hijack.undo_hijack(clip)
                    raise error
                return ([[cond, {}]], )
        
        result = run()
        # print("cond (+)" if multi_conditioning else "uncond (-)",  result[0][0][0]) # debug
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