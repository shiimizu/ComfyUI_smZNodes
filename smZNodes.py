from __future__ import annotations
import comfy
import torch
from typing import List, Tuple
from functools import partial
from .modules import prompt_parser, shared, devices
from .modules.shared import opts, opts_default
from .modules.sd_samplers_cfg_denoiser import CFGDenoiser
from .modules.sd_hijack_clip import FrozenCLIPEmbedderForSDXLWithCustomWords
from .modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedder2WithCustomWords
from .modules.textual_inversion.textual_inversion import Embedding
import comfy.sdxl_clip
import comfy.sd1_clip
import comfy.sample
import comfy.utils
from comfy.sd1_clip import SD1Tokenizer, unescape_important, escape_important, token_weights, expand_directory_list
from nodes import CLIPTextEncode
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from comfy import model_management
import inspect
from textwrap import dedent, indent
import functools
import tempfile
import importlib
import sys
import os
import re
import itertools
import binascii
import math
import copy

try:
    from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
except Exception as err:
    print(f"[smZNodes]: Your ComfyUI version is outdated. Please update to the latest version. ({err})")
    class CLIPTextEncodeSDXL(CLIPTextEncode): ...
    class CLIPTextEncodeSDXLRefiner(CLIPTextEncode): ...

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

class PopulateVars:
    def populate_self_variables(self, from_):
        super_attrs = vars(from_)
        self_attrs = vars(self)
        self_attrs.update(super_attrs)

should_use_fp16_signature = inspect.signature(comfy.model_management.should_use_fp16)
class ClipTextEncoderCustom:

    def _forward(self: comfy.sd1_clip.SDClipModel, tokens):
        def set_embeddings_dtype(dtype = torch.float16, newv = False):
            # dtype_num = lambda d : int(re.sub(r'.*?(\d+)', r'\1', repr(d)))
            _p = should_use_fp16_signature.parameters
            # newer versions of ComfyUI upcasts the transformer embeddings, which is technically correct
            # when it's a newer version, we want to downcast it if we aren't running with --force-fp32
            # newv = 'device' in _p and 'prioritize_performance' in _p # comment this to have default comfy behaviour
            # if dtype_num(dtype) >= 32:
            #     newv = False
            if not newv: return
            # dtype = devices.dtype if dtype != devices.dtype else dtype
            # self.transformer.text_model.embeddings.position_embedding.to(dtype)
            # self.transformer.text_model.embeddings.token_embedding.to(dtype)
            if hasattr(self, 'inner_name'):
                inner_model = getattr(self.transformer, self.inner_name, None)
            else:
                inner_model = None
            if inner_model is not None and hasattr(inner_model, "embeddings"):
                inner_model.embeddings.to(dtype)
            else:
                self.transformer.set_input_embeddings(self.transformer.get_input_embeddings().to(dtype))
            try: self.transformer.to(dtype=dtype)
            except Exception as e: 
                if opts.debug: print(e)
            try: self.transformer.text_model.to(dtype=dtype)
            except Exception as e: 
                if opts.debug: print(e)
        def reset_embeddings_dtype():
            # token_embedding_dtype = position_embedding_dtype = torch.float32
            # self.transformer.text_model.embeddings.token_embedding.to(token_embedding_dtype)
            # self.transformer.text_model.embeddings.position_embedding.to(position_embedding_dtype)
            dtype=torch.float32
            if hasattr(self, 'inner_name'):
                inner_model = getattr(self.transformer, self.inner_name, None)
            else:
                inner_model = None
            if inner_model is not None and hasattr(inner_model, "embeddings"):
                inner_model.embeddings.to(dtype=dtype)
            else:
                self.transformer.set_input_embeddings(self.transformer.get_input_embeddings().to(dtype=dtype))
            try: self.transformer.to(dtype=dtype)
            except Exception as e:
                if opts.debug: print(e)
            try: self.transformer.text_model.to(dtype=dtype)
            except Exception as e:
                if opts.debug: print(e)

        enable_compat = False
        if enable_compat: set_embeddings_dtype(newv=enable_compat)
        z, pooled_output = self.forward(tokens)
        if enable_compat: reset_embeddings_dtype()
        return z.to(dtype=torch.float16) if enable_compat else z, pooled_output

    def encode_with_transformers_comfy_(self, tokens: List[List[int]], return_pooled=False):
        tokens_orig = tokens
        try:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            z, pooled = ClipTextEncoderCustom._forward(self.wrapped, tokens) # self.wrapped.encode(tokens)
        except Exception as e:
            z, pooled = ClipTextEncoderCustom._forward(self.wrapped, tokens_orig)

            # z = self.encode_with_transformers__(tokens_bak)
        if z.device != devices.device:
            z = z.to(device=devices.device)
        # if z.dtype != devices.dtype:
        #     z = z.to(dtype=devices.dtype)
        # if pooled.dtype != devices.dtype:
        #     pooled = pooled.to(dtype=devices.dtype)
        z.pooled = pooled
        return (z, pooled) if return_pooled else z

    def encode_with_transformers_comfy(self, tokens: List[List[int]], return_pooled=False) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This function is different from `clip.cond_stage_model.encode_token_weights()`
        in that the tokens are `List[List[int]]`, not including the weights.

        Originally from `sd1_clip.py`: `encode()` -> `forward()`
        '''
        tokens_orig = tokens
        try:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            z, pooled = self.wrapped(tokens) # self.wrapped.encode(tokens)
        except Exception as e:
            z, pooled = self.wrapped(tokens_orig)

            # z = self.encode_with_transformers__(tokens_bak)
        if z.device != devices.device:
            z = z.to(device=devices.device)
        # if z.dtype != devices.dtype:
        #     z = z.to(dtype=devices.dtype)
        # if pooled.dtype != devices.dtype:
        #     pooled = pooled.to(dtype=devices.dtype)
        z.pooled = pooled
        return (z, pooled) if return_pooled else z

class FrozenOpenCLIPEmbedder2WithCustomWordsCustom(FrozenOpenCLIPEmbedder2WithCustomWords, ClipTextEncoderCustom, PopulateVars):
    def __init__(self, wrapped: comfy.sdxl_clip.SDXLClipG, hijack):
        self.populate_self_variables(wrapped.tokenizer_parent)
        super().__init__(wrapped, hijack)
        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = 0
        # Below is safe to do since ComfyUI uses the same CLIP model
        # for Open Clip instead of an actual Open Clip model?
        self.token_mults = {}
        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab.get(',</w>', None)
        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1
            if mult != 1.0:
                self.token_mults[ident] = mult

    def tokenize_line(self, line):
        line = parse_and_register_embeddings(self, line)
        return super().tokenize_line(line)

    def encode(self, tokens):
        return self.encode_with_transformers(tokens, True)

    def encode_with_transformers(self, tokens, return_pooled=False):
        return self.encode_with_transformers_comfy_(tokens, return_pooled)

    def encode_token_weights(self, tokens):
        pass

    def tokenize(self, texts):
        # assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'
        tokenized = [self.tokenizer(text)["input_ids"][1:-1] for text in texts]
        return tokenized


class FrozenCLIPEmbedderWithCustomWordsCustom(FrozenCLIPEmbedderForSDXLWithCustomWords, ClipTextEncoderCustom, PopulateVars):
    '''
    Custom class that also inherits a tokenizer to have the `_try_get_embedding()` method.
    '''
    def __init__(self, wrapped: comfy.sd1_clip.SD1ClipModel, hijack):
        self.populate_self_variables(wrapped.tokenizer_parent) # SD1Tokenizer
        # self.embedding_identifier_tokenized = wrapped.tokenizer([self.embedding_identifier])["input_ids"][0][1:-1]
        super().__init__(wrapped, hijack)

    def encode_token_weights(self, tokens):
        pass

    def encode(self, tokens):
        return self.encode_with_transformers(tokens, True)

    def encode_with_transformers(self, tokens, return_pooled=False):
        return self.encode_with_transformers_comfy_(tokens, return_pooled)

    def tokenize_line(self, line):
        line = parse_and_register_embeddings(self, line)
        return super().tokenize_line(line)

    def tokenize(self, texts):
        tokenized = [self.tokenizer(text)["input_ids"][1:-1] for text in texts]
        return tokenized

emb_re_ = r"(embedding:)?(?:({}[\w\.\-\!\$\/\\]+(\.safetensors|\.pt|\.bin)|(?(1)[\w\.\-\!\$\/\\]+|(?!)))(\.safetensors|\.pt|\.bin)?)(?::(\d+\.?\d*|\d*\.\d+))?"

def tokenize_with_weights_custom(self, text:str, return_word_ids=False):
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
    parsed_weights = token_weights(text, 1.0)

    embs = get_valid_embeddings(self.embedding_directory) if self.embedding_directory is not None else []
    embs_str = embs_str + '|' if (embs_str:='|'.join(embs)) else ''
    emb_re = emb_re_.format(embs_str)
    emb_re = re.compile(emb_re, flags=re.MULTILINE | re.UNICODE | re.IGNORECASE)

    #tokenize words
    tokens = []
    for weighted_segment, weight in parsed_weights:
        to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            matches = emb_re.finditer(word)
            last_end = 0
            leftovers=[]
            for _, match in enumerate(matches, start=1):
                start=match.start()
                end=match.end()
                if (fragment:=word[last_end:start]):
                    leftovers.append(fragment)
                ext = ext if (ext:=match.group(4)) else ''
                embedding_sname = embedding_sname if (embedding_sname:=match.group(2)) else ''
                embedding_name = embedding_sname + ext
                if embedding_name:
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if opts.debug:
                            print(f'[smZNodes] using embedding:{embedding_name}')
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                last_end = end
            if (fragment:=word[last_end:]):
                leftovers.append(fragment)
                word_new = ''.join(leftovers)
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])

    #reshape token array to CLIP input size
    batched_tokens = []
    batch = []
    if self.start_token is not None:
        batch.append((self.start_token, 1.0, 0))
    batched_tokens.append(batch)
    for i, t_group in enumerate(tokens):
        #determine if we're going to try and keep the tokens in a single batch
        is_large = len(t_group) >= self.max_word_length

        while len(t_group) > 0:
            if len(t_group) + len(batch) > self.max_length - 1:
                remaining_length = self.max_length - len(batch) - 1
                #break word in two and add end token
                if is_large:
                    batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                    batch.append((self.end_token, 1.0, 0))
                    t_group = t_group[remaining_length:]
                #add end token and pad
                else:
                    batch.append((self.end_token, 1.0, 0))
                    if self.pad_to_max_length:
                        batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                #start new batch
                batch = []
                if self.start_token is not None:
                    batch.append((self.start_token, 1.0, 0))
                batched_tokens.append(batch)
            else:
                batch.extend([(t,w,i+1) for t,w in t_group])
                t_group = []

    #fill last batch
    batch.append((self.end_token, 1.0, 0))
    if self.pad_to_max_length:
        batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))

    if not return_word_ids:
        batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

    return batched_tokens

def get_valid_embeddings(embedding_directory):
    from  builtins import any as b_any
    exts = ['.safetensors', '.pt', '.bin']
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]
    embedding_directory = expand_directory_list(embedding_directory)
    embs = []
    for embd in embedding_directory:
        for root, dirs, files in os.walk(embd, topdown=False):
            for name in files:
                if not b_any(x in os.path.splitext(name)[1] for x in exts): continue
                n = os.path.basename(name)
                for ext in exts: n=n.removesuffix(ext)
                embs.append(re.escape(n))
    embs.sort(key=len, reverse=True)
    return embs

def parse_and_register_embeddings(self: FrozenCLIPEmbedderWithCustomWordsCustom|FrozenOpenCLIPEmbedder2WithCustomWordsCustom, text: str, return_word_ids=False):
    from  builtins import any as b_any
    embedding_directory = self.wrapped.tokenizer_parent.embedding_directory
    embs = get_valid_embeddings(embedding_directory)
    embs_str = '|'.join(embs)
    emb_re = emb_re_.format(embs_str + '|' if embs_str else '')
    emb_re = re.compile(emb_re, flags=re.MULTILINE | re.UNICODE | re.IGNORECASE)
    matches = emb_re.finditer(text)
    for matchNum, match in enumerate(matches, start=1):
        found=False
        ext = ext if (ext:=match.group(4)) else ''
        embedding_sname = embedding_sname if (embedding_sname:=match.group(2)) else ''
        embedding_name = embedding_sname + ext
        if embedding_name:
            embed, _ = self.wrapped.tokenizer_parent._try_get_embedding(embedding_name)
            if embed is not None:
                found=True
                if opts.debug:
                    print(f'[smZNodes] using embedding:{embedding_name}')
                if embed.device != devices.device:
                    embed = embed.to(device=devices.device)
                self.hijack.embedding_db.register_embedding(Embedding(embed, embedding_sname), self)
        if not found:
            print(f"warning, embedding:{embedding_name} does not exist, ignoring")
    out = emb_re.sub(r"\2", text)
    return out

def expand(tensor1, tensor2):
    def adjust_tensor_shape(tensor_small, tensor_big):
        # Calculate replication factor
        # -(-a // b) is ceiling of division without importing math.ceil
        replication_factor = -(-tensor_big.size(1) // tensor_small.size(1))
        
        # Use repeat to extend tensor_small
        tensor_small_extended = tensor_small.repeat(1, replication_factor, 1)
        
        # Take the rows of the extended tensor_small to match tensor_big
        tensor_small_matched = tensor_small_extended[:, :tensor_big.size(1), :]
        
        return tensor_small_matched

    # Check if their second dimensions are different
    if tensor1.size(1) != tensor2.size(1):
        # Check which tensor has the smaller second dimension and adjust its shape
        if tensor1.size(1) < tensor2.size(1):
            tensor1 = adjust_tensor_shape(tensor1, tensor2)
        else:
            tensor2 = adjust_tensor_shape(tensor2, tensor1)
    return (tensor1, tensor2)

def reconstruct_schedules(schedules, step):
    create_reconstruct_fn = lambda _cc: prompt_parser.reconstruct_multicond_batch if type(_cc).__name__ == "MulticondLearnedConditioning" else prompt_parser.reconstruct_cond_batch
    reconstruct_fn = create_reconstruct_fn(schedules)
    return reconstruct_fn(schedules, step)


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs, steps=0, current_step=0, multi=False):
        schedules = token_weight_pairs
        texts = token_weight_pairs
        conds_list = [[(0, 1.0)]]
        from .modules.sd_hijack import model_hijack
        try:
            model_hijack.hijack(self)
            if isinstance(token_weight_pairs, list) and isinstance(token_weight_pairs[0], str):
                if multi:   schedules = prompt_parser.get_multicond_learned_conditioning(model_hijack.cond_stage_model, texts, steps, None, opts.use_old_scheduling)
                else:       schedules = prompt_parser.get_learned_conditioning(model_hijack.cond_stage_model, texts, steps, None, opts.use_old_scheduling)
                cond = reconstruct_schedules(schedules, current_step)
                if type(cond) is tuple:
                    conds_list, cond = cond
                cond.pooled.conds_list = conds_list
                cond.pooled.schedules = schedules
            else:
                # comfy++
                def encode_toks(_token_weight_pairs):
                    zs = []
                    first_pooled = None
                    for batch_chunk in _token_weight_pairs:
                        tokens = [x[0] for x in batch_chunk]
                        multipliers = [x[1] for x in batch_chunk]
                        z = model_hijack.cond_stage_model.process_tokens([tokens], [multipliers])
                        if first_pooled == None:
                            first_pooled = z.pooled
                        zs.append(z)
                    zcond = torch.hstack(zs)
                    zcond.pooled = first_pooled
                    return zcond
                # non-sdxl will be something like: {"l": [[]]}
                if isinstance(token_weight_pairs, dict):
                    token_weight_pairs = next(iter(token_weight_pairs.values()))
                cond = encode_toks(token_weight_pairs)
                cond.pooled.conds_list = conds_list
        finally:
                model_hijack.undo_hijack(model_hijack.cond_stage_model)
        return (cond, cond.pooled)

class SD1ClipModel(ClipTokenWeightEncoder): ...

class SDXLClipG(ClipTokenWeightEncoder): ...

class SDXLClipModel(ClipTokenWeightEncoder):

    def encode_token_weights(self: comfy.sdxl_clip.SDXLClipModel, token_weight_pairs, steps=0, current_step=0, multi=False):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]

        self.clip_g.encode_token_weights_orig = self.clip_g.encode_token_weights
        self.clip_l.encode_token_weights_orig = self.clip_l.encode_token_weights
        self.clip_g.cond_stage_model = self.clip_g
        self.clip_l.cond_stage_model = self.clip_l
        self.clip_g.encode_token_weights = partial(SDXLClipG.encode_token_weights, self.clip_g)
        self.clip_l.encode_token_weights = partial(SD1ClipModel.encode_token_weights, self.clip_l)
        try:
            g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g, steps, current_step, multi)
            l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l, steps, current_step, multi)
            # g_out, g_pooled = SDXLClipG.encode_token_weights(self.clip_g, token_weight_pairs_g, steps, current_step, multi)
            # l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l, steps, current_step, multi)
        finally:
            self.clip_g.encode_token_weights = self.clip_g.encode_token_weights_orig
            self.clip_l.encode_token_weights = self.clip_l.encode_token_weights_orig
            self.clip_g.cond_stage_model = None
            self.clip_l.cond_stage_model = None

        if hasattr(g_pooled, 'schedules') and hasattr(l_pooled, 'schedules'):
            g_pooled.schedules = {"g": g_pooled.schedules, "l": l_pooled.schedules}

        g_out, l_out = expand(g_out, l_out)
        l_out, g_out = expand(l_out, g_out)

        return torch.cat([l_out, g_out], dim=-1), g_pooled

class SDXLRefinerClipModel(ClipTokenWeightEncoder):

    def encode_token_weights(self: comfy.sdxl_clip.SDXLClipModel, token_weight_pairs, steps=0, current_step=0, multi=False):
        self.clip_g.encode_token_weights_orig = self.clip_g.encode_token_weights
        self.clip_g.encode_token_weights = partial(SDXLClipG.encode_token_weights, self.clip_g)
        token_weight_pairs_g = token_weight_pairs["g"]
        try: g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g, steps, current_step, multi)
        finally: self.clip_g.encode_token_weights = self.clip_g.encode_token_weights_orig
        if hasattr(g_pooled, 'schedules'):
            g_pooled.schedules = {"g": g_pooled.schedules}
        return (g_out, g_pooled)

def is_prompt_editing(schedules):
    if schedules == None: return False
    if not isinstance(schedules, dict):
        schedules = {'g': schedules}
    for k,v in schedules.items():
        if type(v) == list:
            if len(v[0]) != 1: return True
        else:
            if len(v.batch[0][0].schedules) != 1: return True
    return False

# ===================================================================
# RNG
from .modules import rng_philox
def randn_without_seed(x, generator=None, randn_source="cpu"):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""
    if randn_source == "nv":
        return torch.asarray(generator.randn(x.size()), device=x.device)
    else:
        if generator is not None and generator.device.type == "cpu":
            return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=devices.cpu, generator=generator).to(device=x.device)
        else:
            return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, generator, randn_source):
        # self.rng = p.rng
        self.generator = generator
        self.randn_source = randn_source

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return randn_without_seed(x, generator=self.generator, randn_source=self.randn_source)

def prepare_noise(latent_image, seed, noise_inds=None, device='cpu'):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    from .modules.shared import opts
    from comfy.sample import np
    def get_generator(seed):
        nonlocal device
        nonlocal opts
        _generator = torch.Generator(device=device)
        generator = _generator.manual_seed(seed)
        if opts.randn_source == 'nv':
            generator = rng_philox.Generator(seed)
        return generator
    generator = generator_eta = get_generator(seed)

    if opts.eta_noise_seed_delta > 0:
        seed = min(int(seed + opts.eta_noise_seed_delta), int(0xffffffffffffffff))
        generator_eta = get_generator(seed)


    # hijack randn_like
    import comfy.k_diffusion.sampling
    comfy.k_diffusion.sampling.torch = TorchHijack(generator_eta, opts.randn_source)

    if noise_inds is None:
        shape = latent_image.size()
        if opts.randn_source == 'nv':
            return torch.asarray(generator.randn(shape), device=devices.cpu)
        else:
            return torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        shape = [1] + list(latent_image.size())[1:]
        if opts.randn_source == 'nv':
            noise = torch.asarray(generator.randn(shape), device=devices.cpu)
        else:
            noise = torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

# ===========================================================

def run(clip: comfy.sd.CLIP, text, parser, mean_normalization,
               multi_conditioning, use_old_emphasis_implementation, with_SDXL,
               ascore, width, height, crop_w, crop_h, target_width, target_height,
               text_g, text_l, steps=1, step=0):
    # clip_clone = clip.clone()
    clip_clone = clip
    if (opts_new := clip_clone.patcher.model_options.get('smZ_opts', None)) is not None:
        for k,v in opts_new.__dict__.items():
            setattr(opts, k, v)
    else:
        for k,v in opts_default.__dict__.items():
            setattr(opts, k, v)
    opts.prompt_mean_norm = mean_normalization
    opts.use_old_emphasis_implementation = use_old_emphasis_implementation
    opts.CLIP_stop_at_last_layers = abs(clip.layer_idx or 1)
    is_sdxl = "SDXL" in type(clip.cond_stage_model).__name__
    if is_sdxl:
        # Prevents tensor shape mismatch
        # This is what comfy does by default
        opts.batch_cond_uncond = True
        
    parser_d = {"full": "Full parser",
            "compel": "Compel parser",
            "A1111": "A1111 parser",
            "fixed attention": "Fixed attention",
            "comfy++": "Comfy++ parser",
            }
    opts.prompt_attention = parser_d.get(parser, "Comfy parser")

    sdxl_params = {}
    if with_SDXL and is_sdxl:
        sdxl_params = {
            "aesthetic_score": ascore, "width": width, "height": height,
            "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width,
            "target_height": target_height, "text_g": text_g, "text_l": text_l
        }
    pooled={}
    if hasattr(comfy.sd1_clip, 'SDTokenizer'):
        SDTokenizer = comfy.sd1_clip.SDTokenizer
    else:
        SDTokenizer = comfy.sd1_clip.SD1Tokenizer
    tokenize_with_weights_orig = SDTokenizer.tokenize_with_weights
    if parser == "comfy":
        SDTokenizer.tokenize_with_weights = tokenize_with_weights_custom
        clip_model_type_name = type(clip.cond_stage_model).__name__ 
        if with_SDXL and is_sdxl:
            if clip_model_type_name== "SDXLClipModel":
                out = CLIPTextEncodeSDXL().encode(clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)
                out[0][0][1]['aesthetic_score'] = sdxl_params['aesthetic_score']
            elif clip_model_type_name == "SDXLRefinerClipModel":
                out = CLIPTextEncodeSDXLRefiner().encode(clip, ascore, width, height, text)
                for item in ['aesthetic_score', 'width', 'height', 'text_g', 'text_l']:
                    sdxl_params.pop(item)
                out[0][0][1].update(sdxl_params)
            else:
                raise NotImplementedError()
        else:
            out = CLIPTextEncode().encode(clip, text)
        SDTokenizer.tokenize_with_weights = tokenize_with_weights_orig
        return out
    else:
        texts = [text]
        create_prompts = lambda txts: prompt_parser.SdConditioning(txts)
        texts = create_prompts(texts)
        if is_sdxl:
            if with_SDXL:
                texts = {"g": create_prompts([text_g]), "l": create_prompts([text_l])}
            else:
                texts = {"g": texts, "l": texts}

        clip_clone.cond_stage_model_orig = clip_clone.cond_stage_model
        clip_clone.cond_stage_model.encode_token_weights_orig = clip_clone.cond_stage_model.encode_token_weights

        def patch_cond_stage_model():
            nonlocal clip_clone
            from .smZNodes import SD1ClipModel, SDXLClipModel, SDXLRefinerClipModel
            ctp = type(clip_clone.cond_stage_model)
            clip_clone.cond_stage_model.tokenizer = clip_clone.tokenizer
            if ctp is comfy.sdxl_clip.SDXLClipModel:
                clip_clone.cond_stage_model.encode_token_weights = SDXLClipModel.encode_token_weights
                clip_clone.cond_stage_model.clip_g.tokenizer = clip_clone.tokenizer.clip_g
                clip_clone.cond_stage_model.clip_l.tokenizer = clip_clone.tokenizer.clip_l
            elif ctp is comfy.sdxl_clip.SDXLRefinerClipModel:
                clip_clone.cond_stage_model.encode_token_weights = SDXLRefinerClipModel.encode_token_weights
                clip_clone.cond_stage_model.clip_g.tokenizer = clip_clone.tokenizer.clip_g
            else:
                clip_clone.cond_stage_model.encode_token_weights = SD1ClipModel.encode_token_weights

        tokens = texts
        if parser == "comfy++":
            SDTokenizer.tokenize_with_weights = tokenize_with_weights_custom
            tokens = clip_clone.tokenize(text)
            SDTokenizer.tokenize_with_weights = tokenize_with_weights_orig
        cond = pooled = None
        patch_cond_stage_model()
        try:
            clip_clone.cond_stage_model.encode_token_weights = partial(clip_clone.cond_stage_model.encode_token_weights, clip_clone.cond_stage_model, steps=steps, current_step=step, multi=multi_conditioning)
            cond, pooled = clip_clone.encode_from_tokens(tokens, True)
        finally:
            clip_clone.cond_stage_model = clip_clone.cond_stage_model_orig
            clip_clone.cond_stage_model.encode_token_weights = clip_clone.cond_stage_model.encode_token_weights_orig
        
        if opts.debug:
            print('[smZNodes] using steps', steps)
        gen_id = lambda : binascii.hexlify(os.urandom(1024))[64:72]
        id=gen_id()
        schedules = getattr(pooled, 'schedules', [[(0, 1.0)]])
        conds_list=pooled.conds_list
        pooled=pooled.to(model_management.intermediate_device())
        pooled = {"pooled_output": pooled, "from_smZ": True, "smZid": id, "conds_list": conds_list, **sdxl_params}
        cond=cond.to(model_management.intermediate_device())
        out = [[cond, pooled]]
        if is_prompt_editing(schedules):
            for x in range(1,steps):
                if type(schedules) is not dict:
                    cond=reconstruct_schedules(schedules, x)
                    if type(cond) is tuple:
                        conds_list, cond = cond
                        pooled['conds_list'] = conds_list
                    cond=cond
                elif type(schedules) is dict and len(schedules) == 1: # SDXLRefiner
                    cond = reconstruct_schedules(next(iter(schedules.values())), x)
                    if type(cond) is tuple:
                        conds_list, cond = cond
                        pooled['conds_list'] = conds_list
                    cond=cond
                elif type(schedules) is dict:
                    g_out = reconstruct_schedules(schedules['g'], x)
                    if type(g_out) is tuple: _, g_out = g_out
                    l_out = reconstruct_schedules(schedules['l'], x)
                    if type(l_out) is tuple: _, l_out = l_out
                    g_out, l_out = expand(g_out, l_out)
                    l_out, g_out = expand(l_out, g_out)
                    cond = torch.cat([l_out, g_out], dim=-1)
                else:
                    raise NotImplementedError
                cond=cond.to(model_management.intermediate_device())
                out = out + [[cond, pooled]]
        out[0][1]['orig_len'] = len(out)
    return (out,)

# ========================================================================

from server import PromptServer
def prompt_handler(json_data):
    data=json_data['prompt']
    def tmp():
        nonlocal data
        current_clip_id = None
        def find_nearest_ksampler(clip_id):
            """Find the nearest KSampler node that references the given CLIPTextEncode id."""
            for ksampler_id, node in data.items():
                if "Sampler" in node["class_type"] or "sampler" in node["class_type"]:
                    # Check if this KSampler node directly or indirectly references the given CLIPTextEncode node
                    if check_link_to_clip(ksampler_id, clip_id):
                        return get_steps(data, ksampler_id)
            return None

        def get_steps(graph, node_id):
            node = graph.get(str(node_id), {})
            steps_input_value = node.get("inputs", {}).get("steps", None)
            if steps_input_value is None:
                steps_input_value = node.get("inputs", {}).get("sigmas", None)

            while(True):
                # Base case: it's a direct value
                if is_number(steps_input_value):
                    return min(max(1, int(steps_input_value)), 10000)

                # Loop case: it's a reference to another node
                elif isinstance(steps_input_value, list):
                    ref_node_id, ref_input_index = steps_input_value
                    ref_node = graph.get(str(ref_node_id), {})
                    steps_input_value = ref_node.get("inputs", {}).get("steps", None)
                    if steps_input_value is None:
                        keys = list(ref_node.get("inputs", {}).keys())
                        ref_input_key = keys[ref_input_index % len(keys)]
                        steps_input_value = ref_node.get("inputs", {}).get(ref_input_key)
                else:
                    return None

        def check_link_to_clip(node_id, clip_id, visited=None):
            """Check if a given node links directly or indirectly to a CLIPTextEncode node."""
            if visited is None:
                visited = set()

            node = data[node_id]
            
            if node_id in visited:
                return False
            visited.add(node_id)

            for input_value in node["inputs"].values():
                if isinstance(input_value, list) and input_value[0] == clip_id:
                    return True
                if isinstance(input_value, list) and check_link_to_clip(input_value[0], clip_id, visited):
                    return True

            return False

        def is_number(s):
            try:
                float(s)
                return True
            except (ValueError, TypeError):
                return False

        # Update each CLIPTextEncode node's steps with the steps from its nearest referencing KSampler node
        for clip_id, node in data.items():
            if node["class_type"] == "smZ CLIPTextEncode":
                current_clip_id = clip_id
                steps = find_nearest_ksampler(clip_id)
                if steps is not None:
                    node["inputs"]["smZ_steps"] = steps
                    if opts.debug:
                        print(f'[smZNodes] id: {current_clip_id} | steps: {steps}')
    tmp()
    return json_data

if hasattr(PromptServer.instance, 'add_on_prompt_handler'):
    PromptServer.instance.add_on_prompt_handler(prompt_handler)

# ========================================================================
def bounded_modulo(number, modulo_value):
    return number if number < modulo_value else modulo_value

def get_adm(c):
    for y in ["adm_encoded", "c_adm", "y"]:
        if y in c:
            c_c_adm = c[y]
            if y == "adm_encoded": y="c_adm"
            if type(c_c_adm) is not torch.Tensor: c_c_adm = c_c_adm.cond 
            return {y: c_c_adm, 'key': y}
    return None

getp=lambda x: x[1] if type(x) is list else x
def get_cond(c, current_step):
    """Group by smZ conds that may do prompt-editing / regular conds / comfy conds."""
    _cond = []
    # Group by conds from smZ
    fn=lambda x : getp(x).get("from_smZ", None) is not None
    an_iterator = itertools.groupby(c, fn )
    for key, group in an_iterator:
        ls=list(group)
        # Group by prompt-editing conds
        fn2=lambda x : getp(x).get("smZid", None)
        an_iterator2 = itertools.groupby(ls, fn2)
        for key2, group2 in an_iterator2:
            ls2=list(group2)
            if key2 is not None:
                orig_len = getp(ls2[0]).get('orig_len', 1)
                i = bounded_modulo(current_step, orig_len - 1)
                _cond = _cond + [ls2[i]]
            else:
                _cond = _cond + ls2
    return _cond

CFGNoisePredictorOrig = comfy.samplers.CFGNoisePredictor
class CFGNoisePredictor(CFGNoisePredictorOrig):
    def __init__(self, model):
        super().__init__(model)
        self.step = 0
        self.inner_model2 = CFGDenoiser(model.apply_model)
        self.s_min_uncond = opts.s_min_uncond
        self.c_adm = None
        self.init_cond = None
        self.init_uncond = None
        self.is_prompt_editing_u = False
        self.is_prompt_editing_c = False

    def apply_model(self, *args, **kwargs):
        x=kwargs['x'] if 'x' in kwargs else args[0]
        timestep=kwargs['timestep'] if 'timestep' in kwargs else args[1]
        cond=kwargs['cond'] if 'cond' in kwargs else args[2]
        uncond=kwargs['uncond'] if 'uncond' in kwargs else args[3]
        cond_scale=kwargs['cond_scale'] if 'cond_scale' in kwargs else args[4]
        model_options=kwargs['model_options'] if 'model_options' in kwargs else {}

        cc=get_cond(cond, self.step)
        uu=get_cond(uncond, self.step)
        self.step += 1

        if (any([getp(p).get('from_smZ', False) for p in cc]) or 
            any([getp(p).get('from_smZ', False) for p in uu])):
            if model_options.get('transformer_options',None) is None:
                model_options['transformer_options'] = {}
            model_options['transformer_options']['from_smZ'] = True

        if not opts.use_CFGDenoiser or not model_options['transformer_options'].get('from_smZ', False):
            if 'cond' in kwargs: kwargs['cond'] = cc
            else: args[2]=cc
            if 'uncond' in kwargs: kwargs['uncond'] = uu
            else: args[3]=uu
            out = super().apply_model(*args, **kwargs)
        else:
            # Only supports one cond
            for ix in range(len(cc)):
                if getp(cc[ix]).get('from_smZ', False):
                    cc = [cc[ix]]
                    break
            for ix in range(len(uu)):
                if getp(uu[ix]).get('from_smZ', False):
                    uu = [uu[ix]]
                    break
            c=getp(cc[0])
            u=getp(uu[0])
            _cc = cc[0][0] if type(cc[0]) is list else cc[0]['model_conds']['c_crossattn'].cond
            _uu = uu[0][0] if type(uu[0]) is list else uu[0]['model_conds']['c_crossattn'].cond
            conds_list = c.get('conds_list', [[(0, 1.0)]])
            if 'model_conds' in c: c = c['model_conds']
            if 'model_conds' in u: u = u['model_conds']
            c_c_adm = get_adm(c)
            if c_c_adm is not None:
                u_c_adm = get_adm(u)
                k = c_c_adm['key']
                self.c_adm = {k: torch.cat([c_c_adm[k], u_c_adm[u_c_adm['key']]]).to(device=x.device), 'key': k}
                # SDXL. Need to pad with repeats
                _cc, _uu = expand(_cc, _uu)
                _uu, _cc = expand(_uu, _cc)
            x.c_adm = self.c_adm
            image_cond = txt2img_image_conditioning(None, x)
            out = self.inner_model2(x, timestep, cond=(conds_list, _cc), uncond=_uu, cond_scale=cond_scale, s_min_uncond=self.s_min_uncond, image_cond=image_cond)
        return out

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

# =======================================================================================

def inject_code(original_func, data):
    # Get the source code of the original function
    original_source = inspect.getsource(original_func)

    # Split the source code into lines
    lines = original_source.split("\n")

    for item in data:
        # Find the line number of the target line
        target_line_number = None
        for i, line in enumerate(lines):
            if item['target_line'] in line:
                target_line_number = i + 1

                # Find the indentation of the line where the new code will be inserted
                indentation = ''
                for char in line:
                    if char == ' ':
                        indentation += char
                    else:
                        break
                
                # Indent the new code to match the original
                code_to_insert = dedent(item['code_to_insert'])
                code_to_insert = indent(code_to_insert, indentation)
                break

        if target_line_number is None:
            raise FileNotFoundError
            # Target line not found, return the original function
            # return original_func

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
