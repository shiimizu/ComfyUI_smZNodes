
from comfy.sd import CLIP
from comfy.sd1_clip import SD1ClipModel
from types import MethodType
from .modules.shared import opts
import torch

def encode_from_tokens_with_custom_mean(clip: CLIP, tokens, return_pooled=False):
    '''
    The function is our rendition of `clip.encode_from_tokens()`.
    It still calls `clip.encode_from_tokens()` but hijacks the 
    `clip.cond_stage_model.encode_token_weights()` method
    so we can run our own version of `encode_token_weights()`

    Originally from `sd.py`: `encode_from_tokens()`
    '''
    ret = None
    encode_token_weights_backup = clip.cond_stage_model.encode_token_weights
    try:
        clip.cond_stage_model.encode_token_weights = MethodType(encode_token_weights_customized, clip.cond_stage_model)
        ret = clip.encode_from_tokens(tokens, return_pooled)
        clip.cond_stage_model.encode_token_weights = encode_token_weights_backup
    except Exception as error:
        clip.cond_stage_model.encode_token_weights = encode_token_weights_backup
        raise error

    return ret

def encode_token_weights_customized(self: SD1ClipModel, token_weight_pairs):
    to_encode = list(self.empty_tokens)
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        weights = list(map(lambda a: a[1], x))
        to_encode.append(tokens)

    out, pooled = self.encode(to_encode)
    zw = torch.asarray(weights).to(device=out.device)

    z_empty = out[0:1]
    if pooled.shape[0] > 1:
        first_pooled = pooled[1:2]
    else:
        first_pooled = pooled[0:1]

    output = []
    for i in range(1, out.shape[0]):
        # 3D -> 2D
        z = out[i:i+1]
        batch_multipliers = zw[i:i+1]
        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        if opts.prompt_mean_norm:
            original_mean = z.mean()
            z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
            new_mean = z.mean()
            z = z * (original_mean / new_mean)
        else:
            z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
            # for i in range(len(z)):
            #     for j in range(len(z[i])):
            #         weight = token_weight_pairs[i - 1][j][1]
            #         z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
        output.append(z)

    if (len(output) == 0):
        return z_empty, first_pooled
    return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()
