import comfy
import torch
from comfy.sd import CLIP
from typing import List, Tuple
from types import MethodType
from functools import partial
from .modules import prompt_parser, shared, devices
from .modules.shared import opts
from .modules.sd_samplers_kdiffusion import CFGDenoiser
from .modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords
from comfy.sd1_clip import SD1Tokenizer, SD1ClipModel, unescape_important, escape_important
from comfy.sdxl_clip import SDXLTokenizer, SDXLClipModel
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from types import MethodType
import nodes
import inspect
from textwrap import dedent
import functools
import tempfile
import importlib
import sys


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
    if type(self) == FrozenCLIPEmbedderWithCustomWordsCustom:
        to_encode = list(self.wrapped.empty_tokens)
    else:
        to_encode = list(self.empty_tokens)
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        weights = list(map(lambda a: a[1], x))
        to_encode.append(tokens)

    if type(self) == FrozenCLIPEmbedderWithCustomWordsCustom:
        out, pooled = self.encode_with_transformers(to_encode, return_pooled=True)
    else:
        out, pooled = self.encode(to_encode)
    zw = torch.asarray(weights).to(device=out.device)

    z_empty = out[0:1]
    if pooled.shape[0] > 1:
        first_pooled = pooled[1:2]
    else:
        first_pooled = pooled[0:1]

    output = []
    for k in range(1, out.shape[0]):
        # 3D -> 2D
        z = out[k:k+1]
        batch_multipliers = zw[k:k+1]
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
            #         weight = token_weight_pairs[k - 1][j][1]
            #         z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
        output.append(z)

    if (len(output) == 0):
        return z_empty, first_pooled
    return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()

def get_learned_conditioning(self, c):
    if self.cond_stage_forward is None:
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(c)
    else:
        assert hasattr(self.cond_stage_model, self.cond_stage_forward)
        c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
    return c

# SDXLTokenizer inherits SD1Tokenizer
class FrozenCLIPEmbedderWithCustomWordsCustom(FrozenCLIPEmbedderWithCustomWords, SDXLTokenizer):
    '''
    Custom `FrozenCLIPEmbedderWithCustomWords` class that also inherits
    `SD1Tokenizer` to have the `_try_get_embedding()` method.

    Supports SD1.x and SDXL and may not support SD2.x.
    '''
    def populate_self_variables(self, from_):
        super_attrs = vars(from_)
        self_attrs = vars(self)
        self_attrs.update(super_attrs)

    def __init__(self, wrapped: SD1ClipModel, hijack, clip_type=""):
        self.clip_type = clip_type
        # okay to modiy since CLIP was cloned
        if type(hijack.clip_orig.tokenizer) == SDXLTokenizer:
            self.populate_self_variables(hijack.clip_orig.tokenizer)
            if clip_type == "clip_g":
                self.populate_self_variables(hijack.clip_orig.tokenizer.clip_g)
                wrapped.tokenizer = hijack.clip_orig.tokenizer.clip_g.tokenizer
            elif clip_type == "clip_l":
                self.populate_self_variables(hijack.clip_orig.tokenizer.clip_l)
                wrapped.tokenizer = hijack.clip_orig.tokenizer.clip_l.tokenizer
        else:
            self.populate_self_variables(hijack.clip_orig.tokenizer) # SD1Tokenizer
            wrapped.tokenizer = hijack.clip_orig.tokenizer.tokenizer # CLIPTokenizer.from_pretrained(tokenizer_path)
        # self.embedding_identifier_tokenized = hijack.clip_orig.tokenizer.tokenizer([self.embedding_identifier])["input_ids"][0][1:-1]
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens, return_pooled=False):
        return self.encode_from_tokens_comfy(tokens, return_pooled)

    def encode_with_transformers_comfy(self, tokens: List[List[int]], return_pooled=False) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This function is different from `clip.cond_stage_model.encode_token_weights()`
        in that the tokens are `List[List[int]]`, not including the weights.

        Originally from `sd1_clip.py`: `encode()` -> `forward()`
        '''
        if type(tokens) == torch.Tensor:
            tokens = tokens.tolist()
        z, pooled = self.wrapped(tokens) # self.wrapped.encode(tokens)
        return (z, pooled) if return_pooled else z

    def encode_from_tokens_comfy(self, tokens: List[List[int]], return_pooled=False):
        '''
        The function is our rendition of `clip.encode_from_tokens()`.
        It still calls `clip.encode_from_tokens()` but hijacks the
        `clip.cond_stage_model.encode_token_weights()` method
        so we can run our own version of `encode_token_weights()`

        Originally from `sd.py`: `encode_from_tokens()`
        '''
        # note:
        # self.wrapped = self.hijack.clip_orig.cond_stage_model
        ret = None
        if type(tokens) == torch.Tensor:
            tokens = tokens.tolist()
        encode_token_weights_backup = self.hijack.clip_orig.cond_stage_model.encode_token_weights
        try:
            self.hijack.clip_orig.cond_stage_model.encode_token_weights = MethodType(self.encode_with_transformers_comfy, tokens)
            ret = self.hijack.clip_orig.encode_from_tokens(tokens, return_pooled)
            self.hijack.clip_orig.cond_stage_model.encode_token_weights = encode_token_weights_backup
        except Exception as error:
            self.hijack.clip_orig.cond_stage_model.encode_token_weights = encode_token_weights_backup
            raise error
        return ret

    def tokenize_line(self, line):
        self.parse_and_register_embeddings(line) # register embeddings, discard return
        return super().tokenize_line(line)

    # This function has been added to apply embeddings
    # from sd1_clip.py @ tokenize_with_weights()
    def parse_and_register_embeddings(self, text: str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        # parsed_weights = token_weights(text, 1.0)
        parsed = prompt_parser.parse_prompt_attention(text)
        parsed_weights = [tuple(tw) for tw in parsed]

        #tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            # to_tokenize = [x for x in weighted_segment if x != ""]

            tmp=[]
            # print(word)
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                emb_idx = word.find(self.embedding_identifier)
                # word.startswith(self.embedding_identifier)
                if emb_idx != -1 and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')

                    embedding_name_verbose = word[emb_idx:].strip('\n')
                    embedding_name = word[emb_idx+len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name.strip())

                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        embed = embed.to(device=devices.device)
                        if self.clip_type == "clip_g" or self.clip_type == "clip_l":
                            # backup_tokenize = self.hijack.cond_stage_model.tokenize
                            if not hasattr(self.hijack.cond_stage_model, "tokenize") or self.hijack.cond_stage_model.tokenize == None:
                                if self.clip_type == "clip_g":
                                    self.hijack.cond_stage_model.tokenize = self.hijack.cond_stage_model.clip_g.tokenize
                                elif self.clip_type == "clip_l":
                                    self.hijack.cond_stage_model.tokenize = self.hijack.cond_stage_model.clip_l.tokenize
                            self.hijack.embedding_db.register_embedding(Embedding(embed, embedding_name_verbose), self.hijack)
                            # self.hijack.cond_stage_model.tokenize = backup_tokenize
                            self.hijack.cond_stage_model.tokenize = None
                        else:
                            self.hijack.embedding_db.register_embedding(Embedding(embed, embedding_name_verbose), self.hijack)
                        if len(embed.shape) == 1:
                            # tokens.append([(embed, weight)])
                            tmp += [(embed, weight)]
                        else:
                            # tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                            tmp += [(embed[x], weight) for x in range(embed.shape[0])]
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                # tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][1:-1]])
                tmp += [(word, weight)]
                # tokens.append(tmp)
                tokens += tmp
        return tokens

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        sends one single prompt chunk to be encoded by transformers neural network.
        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
        corresponds to one token.
        """
        try:
            tokens = torch.asarray(remade_batch_tokens).to(devices.device)
            # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
            if self.id_end != self.id_pad:
                for batch_pos in range(len(remade_batch_tokens)):
                    index = remade_batch_tokens[batch_pos].index(self.id_end)
                    tokens[batch_pos, index+1:tokens.shape[1]] = self.id_pad
        except:
            # comfy puts embeddings into the tokens list and torch.asarray will give an error, so we do this
            tokens = remade_batch_tokens
        z = self.encode_with_transformers(tokens).to(devices.device)
        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(batch_multipliers).to(devices.device)
        if opts.prompt_mean_norm:
            original_mean = z.mean()
            z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
            new_mean = z.mean()
            z *= (original_mean / new_mean)
        else:
            z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        return z

class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum

def encode_from_texts(clip: CLIP, texts, steps = 1, return_pooled=False, multi=False):
    '''
    The function is our rendition of `clip.encode_from_tokens()`.
    It still calls `clip.encode_from_tokens()` but hijacks the
    `clip.cond_stage_model.encode_token_weights()` method
    so we can run our own version of `encode_token_weights()`

    Originally from `sd.py`: `encode_from_tokens()`
    '''
    ret = None
    clip_clone = clip
    tokens=texts
    if type(clip.cond_stage_model) == SDXLClipModel:
        clip = clip.cond_stage_model.clip_g.hijack.clip_orig # doesn't seem to work
        # if type(clip.cond_stage_model.clip_g).__name__ == "FrozenCLIPEmbedderWithCustomWordsCustom":
        #     encode_token_weights_backup_clip_g = clip.cond_stage_model.clip_g.wrapped.encode_token_weights
        #     encode_token_weights_backup_clip_l = clip.cond_stage_model.clip_l.wrapped.encode_token_weights
        #     try:
        #         partial_method = partial(get_learned_conditioning_custom, steps=steps, return_pooled=return_pooled, multi=multi)
        #         clip.cond_stage_model.clip_g.wrapped.encode_token_weights = MethodType(partial_method, clip_clone.cond_stage_model.clip_g)
        #         clip.cond_stage_model.clip_l.wrapped.encode_token_weights = MethodType(partial_method, clip_clone.cond_stage_model.clip_l)
        #         ret = clip.encode_from_tokens(tokens, return_pooled)
        #         clip.cond_stage_model.clip_g.wrapped.encode_token_weights = encode_token_weights_backup_clip_g
        #         clip.cond_stage_model.clip_l.wrapped.encode_token_weights = encode_token_weights_backup_clip_l
        #     except Exception as error:
        #         clip.cond_stage_model.clip_g.wrapped.encode_token_weights = encode_token_weights_backup_clip_g
        #         clip.cond_stage_model.clip_l.wrapped.encode_token_weights = encode_token_weights_backup_clip_l
        #         raise error     
        # else:
        if type(clip.cond_stage_model.clip_g).__name__ == "FrozenCLIPEmbedderWithCustomWordsCustom":
            encode_token_weights_backup_clip_g = clip.cond_stage_model.clip_g.wrapped.encode_token_weights
            encode_token_weights_backup_clip_l = clip.cond_stage_model.clip_l.wrapped.encode_token_weights
        else:
            encode_token_weights_backup_clip_g = clip.cond_stage_model.clip_g.encode_token_weights
            encode_token_weights_backup_clip_l = clip.cond_stage_model.clip_l.encode_token_weights
        try:
            partial_method = partial(get_learned_conditioning_custom, steps=steps, return_pooled=return_pooled, multi=multi)
            clip.cond_stage_model.clip_g.encode_token_weights = MethodType(partial_method, clip_clone.cond_stage_model.clip_g)
            clip.cond_stage_model.clip_l.encode_token_weights = MethodType(partial_method, clip_clone.cond_stage_model.clip_l)
            ret = clip.encode_from_tokens(tokens, return_pooled)
            clip.cond_stage_model.clip_g.encode_token_weights = encode_token_weights_backup_clip_g
            clip.cond_stage_model.clip_l.encode_token_weights = encode_token_weights_backup_clip_l
        except Exception as error:
            clip.cond_stage_model.clip_g.encode_token_weights = encode_token_weights_backup_clip_g
            clip.cond_stage_model.clip_l.encode_token_weights = encode_token_weights_backup_clip_l
            raise error
    else:
        clip = clip.cond_stage_model.hijack.clip_orig
        encode_token_weights_backup = clip.cond_stage_model.encode_token_weights
        try:
            partial_method = partial(get_learned_conditioning_custom, steps=steps, return_pooled=return_pooled, multi=multi)
            clip.cond_stage_model.encode_token_weights = MethodType(partial_method, clip_clone.cond_stage_model)
            ret = clip.encode_from_tokens(tokens, return_pooled)
            clip.cond_stage_model.encode_token_weights = encode_token_weights_backup
        except Exception as error:
            clip.cond_stage_model.encode_token_weights = encode_token_weights_backup
            raise error

    return ret

# tokenize then encode
# from prompt_parser.py: get_learned_conditioning()
def get_learned_conditioning_custom(model: FrozenCLIPEmbedderWithCustomWordsCustom, prompts: List[str], steps = 1, return_pooled=True, multi=False):
    if multi:
        res_indexes, prompt_flat_list, _prompt_indexes = prompt_parser.get_multicond_prompt_list(prompts)
        prompts = prompt_flat_list
    res = []
    prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue
        texts = [x[1] for x in prompt_schedule]
        # forward function
        # conds = model.get_learned_conditioning(texts)
        cond, pooled = forward_custom(model, texts)

        # there's only one prompt_schedule since steps = 1, and prompts with special syntax are not processed
        # with prompt schedules., i.e those <text>:from:to. That's the job of the sampler. So we can return early here.
        return (cond, pooled) if return_pooled else cond

        # cond_schedule = []
        # for i, (end_at_step, _text) in enumerate(prompt_schedule):
        #     cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))
        # cache[prompt] = cond_schedule
        # res.append(cond_schedule)
        # res += tokens
    # return res # [res]


# This function is from the forward() function of FrozenCLIPEmbedderWithCustomWordsBase
def forward_custom(self: FrozenCLIPEmbedderWithCustomWordsCustom, texts: List[str]) -> List[List[Tuple[int, float]]]:
    batch_chunks, _token_count = self.process_texts(texts)
    used_embeddings = {}
    chunk_count = max([len(x) for x in batch_chunks])
    zs = []
    all_twp = [] # added by me
    for i in range(chunk_count):
        batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]
        tokens = [x.tokens for x in batch_chunk]
        multipliers = [x.multipliers for x in batch_chunk]
        self.hijack.fixes = [x.fixes for x in batch_chunk]
        for fixes in self.hijack.fixes:
            for _position, embedding in fixes:
                used_embeddings[embedding.name] = embedding
        z = self.process_tokens(tokens, multipliers)
        zs.append(z)
        all_twp += [list(zip(x.tokens, x.multipliers)) for x in batch_chunk] # added by me
    if len(used_embeddings) > 0:
        embeddings_list = ", ".join([f'{name} [{embedding.checksum()}]' for name, embedding in used_embeddings.items()])
        self.hijack.comments.append(f"Used embeddings: {embeddings_list}")
    # added by me ============================================
    ret = torch.hstack(zs).cpu()

    # Instead of encoding individual tokens, we get all tokens, then encode all of them at once, like comfy.
    cond, pooled = encode_token_weights_customized(self, all_twp)

    if opts.use_old_emphasis_implementation:
        from .modules import sd_hijack_clip_old
        ret = sd_hijack_clip_old.forward_old(self, texts).cpu()

    return (ret, pooled) # ret has correct applied mean, not cond. But pooled was from all the tokens, which is correct.

# =======================================================================================

def inject_code(original_func, target_line, code_to_insert):
    # Get the source code of the original function
    original_source = inspect.getsource(original_func)

    # Split the source code into lines
    lines = original_source.split("\n")

    # Find the line number of the target line
    target_line_number = None
    for i, line in enumerate(lines):
        if target_line in line:
            target_line_number = i + 1
            break

    if target_line_number is None:
        # Target line not found, return the original function
        return original_func

    # Insert the code to be injected after the target line
    lines.insert(target_line_number, code_to_insert)

    # Recreate the modified source code
    modified_source = "\n".join(lines)
    modified_source = dedent(modified_source.strip("\n"))

    # Create a temporary file to write the modified source code so I can still view the
    # source code when debugging.
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
        temp_file.write(modified_source)
        temp_file.flush()

        MODULE_PATH = temp_file.name
        MODULE_NAME = __name__.split('.')[0] + "_patch_modules"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Pass global variables to the modified module
        globals_dict = original_func.__globals__
        for key, value in globals_dict.items():
            setattr(module, key, value)
        modified_module = module

        # Retrieve the modified function from the module
        modified_function = getattr(modified_module, original_func.__name__)

    # If the original function was a method, bind it to the first argument (self)
    if inspect.ismethod(original_func):
        modified_function = modified_function.__get__(original_func.__self__, original_func.__class__)

    # Update the metadata of the modified function to associate it with the original function
    functools.update_wrapper(modified_function, original_func)

    # Return the modified function
    return modified_function

# ========================================================================

def txt2img_image_conditioning(sd_model, x, width=None, height=None):
    return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)
    # if sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models
    #     # The "masked-image" in this case will just be all zeros since the entire image is masked.
    #     image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
    #     image_conditioning = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image_conditioning))
    #     # Add the fake full 1s mask to the first dimension.
    #     image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
    #     image_conditioning = image_conditioning.to(x.dtype)
    #     return image_conditioning
    # elif sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models
    #     return x.new_zeros(x.shape[0], 2*sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)
    # else:
    #     # Dummy zero conditioning if we're not using inpainting or unclip models.
    #     # Still takes up a bit of memory, but no encoder call.
    #     # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
    #     return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)

class KSamplerX0Inpaint_smZ(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.inner_model = model
        self.inner_model = CFGDenoiser(model)
        self.s_min_uncond = 0.0 # getattr(p, 's_min_uncond', 0.0)
        self.inner_model.device = comfy.model_management.get_torch_device()
        self.inner_model.sigma_to_t = model.sigma_to_t
        self.inner_model.t_to_sigma = model.t_to_sigma
        self.inner_model.sigmas = model.sigmas

    # cond_concat, model_option is used in comfy.samplers.sampling_function
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, cond_concat=None, model_options={}, seed=None):
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + (self.latent_image + self.noise * sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1))) * latent_mask
        # out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, cond_concat=cond_concat, model_options=model_options, seed=seed)
        image_cond = txt2img_image_conditioning(self, x)
        out = self.inner_model(x, sigma, cond=cond[0][0], uncond=uncond[0][0], cond_scale=cond_scale, s_min_uncond=self.s_min_uncond, image_cond=image_cond)

        if denoise_mask is not None:
            out *= denoise_mask

        if denoise_mask is not None:
            out += self.latent_image * latent_mask
        return out

def set_model_k(self):
    self.model_denoise = self.model # main change
    if self.model.parameterization == "v":
        self.model_wrap = comfy.samplers.CompVisVDenoiser(self.model_denoise, quantize=True)
    else:
        self.model_wrap = comfy.samplers.k_diffusion_external.CompVisDenoiser(self.model_denoise, quantize=True)
    self.model_wrap.parameterization = self.model.parameterization
    self.model_k = KSamplerX0Inpaint_smZ(self.model_wrap)

class SDKSampler(comfy.samplers.KSampler):
    def __init__(self, *args, **kwargs):
        super(SDKSampler, self).__init__(*args, **kwargs)
        if opts.use_CFGDenoiser:
            set_model_k(self)

# Custom KSampler using CFGDenoiser. Unused.
class smz_SDKSampler(nodes.KSampler):
    @classmethod
    def INPUT_TYPES(s):
        it = super(smz_SDKSampler, s).INPUT_TYPES()
        if not it.get('hidden'):
            it['hidden'] = {}
        it['hidden']['use_CFGDenoiser'] = ([False, True],{"default": True})
        return it
    def sample(self, use_CFGDenoiser=True, *args, **kwargs):
        opts.data['use_CFGDenoiser'] = use_CFGDenoiser
        backup_KSampler = comfy.samplers.KSampler
        ret = None
        try:
            comfy.samplers.KSampler = SDKSampler
            ret = super(smz_SDKSampler, self).sample(*args, **kwargs)
            comfy.samplers.KSampler = backup_KSampler
        except Exception as err:
            comfy.samplers.KSampler = backup_KSampler
            raise err
        return ret

# ========================================================================

# DPM++ 2M alt

from tqdm.auto import trange
@torch.no_grad()
def sample_dpmpp_2m_alt(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + (0.15 * (sigma_progress * sigma_progress))
        old_denoised = denoised * adjustment_factor
    return x


def add_sample_dpmpp_2m_alt():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if "dpmpp_2m_alt" not in KSampler.SAMPLERS:
        try:
            idx = KSampler.SAMPLERS.index("dpmpp_2m")
            KSampler.SAMPLERS.insert(idx+1, "dpmpp_2m_alt")
            setattr(k_diffusion_sampling, 'sample_dpmpp_2m_alt', sample_dpmpp_2m_alt)
            import importlib
            importlib.reload(k_diffusion_sampling)
        except ValueError as err:
            pass