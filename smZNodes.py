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
from comfy.sd1_clip import unescape_important, escape_important, token_weights, expand_directory_list
from nodes import CLIPTextEncode
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from comfy.sample import np
from comfy import model_management
import comfy.samplers
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
from copy import deepcopy

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

should_use_fp16_signature = inspect.signature(model_management.should_use_fp16)
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

emb_re_ = r"(embedding:)?(?:({}[\w\.\-\!\$\/\\]+(\.safetensors|\.pt|\.bin)|(?(1)[\w\.\-\!\$\/\\]+|(?!)))(\.safetensors|\.pt|\.bin)?)(?:(:)(\d+\.?\d*|\d*\.\d+))?"

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
                ext = (match.group(4) or (match.group(3) or ''))
                embedding_sname = (match.group(2) or '').removesuffix(ext)
                embedding_name = embedding_sname + ext
                embedding_name = embedding_name.replace('\\', '/') if '\\' in embedding_name else embedding_name
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
                tokens.append([(t, weight) for t in self.tokenizer(word_new)["input_ids"][self.tokens_start:-1]])

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

def get_valid_embeddings(embedding_directories):
    from  builtins import any as b_any
    exts = ['.safetensors', '.pt', '.bin']
    if isinstance(embedding_directories, str):
        embedding_directories = [embedding_directories]
    embs = set()
    for embd in embedding_directories:
        for root, dirs, files in os.walk(embd, followlinks=True, topdown=False):
            for name in files:
                if not b_any(x in os.path.splitext(name)[1] for x in exts): continue
                n = os.path.basename(name)
                for ext in exts: n=n.removesuffix(ext)
                n = os.path.normpath(os.path.join(os.path.relpath(root, embd), n))
                embs.add(re.escape(n))
                # add its counterpart
                if '/' in n:
                    embs.add(re.escape(n.replace('/', '\\')))
                elif '\\' in n: 
                    embs.add(re.escape(n.replace('\\', '/')))
    embs = sorted(embs, key=len, reverse=True)
    return embs

def parse_and_register_embeddings(self: FrozenCLIPEmbedderWithCustomWordsCustom|FrozenOpenCLIPEmbedder2WithCustomWordsCustom, text: str, return_word_ids=False):
    from  builtins import any as b_any
    text_ = escape_important(text)
    embedding_directories = self.wrapped.tokenizer_parent.embedding_directory
    embs = get_valid_embeddings(embedding_directories)
    embs_str = escape_important('|'.join(embs))
    emb_re = emb_re_.format(embs_str + '|' if embs_str else '')
    emb_re = re.compile(emb_re, flags=re.MULTILINE | re.UNICODE | re.IGNORECASE)
    matches = emb_re.finditer(text_)
    clip_key = "clip_g" if "SDXLClipG" in type(self.wrapped).__name__ else "clip_l"
    if not getattr(self.hijack.embedding_db, 'embeddings', None):
        self.hijack.embedding_db.embeddings = {}
    embeddings = self.hijack.embedding_db.embeddings
    for matchNum, match in enumerate(matches, start=1):
        found=False
        ext = (match.group(4) or (match.group(3) or ''))
        embedding_sname = (match.group(2) or '').removesuffix(ext)
        if '\\' in embedding_sname:
            embedding_sname_new = embedding_sname.replace('\\', '/')
            text_ = text_.replace(embedding_sname, embedding_sname_new)
            embedding_sname = embedding_sname_new
        embedding_name = unescape_important(embedding_sname) + ext
        if embedding_name:
            embed, _ = self.wrapped.tokenizer_parent._try_get_embedding(embedding_name)
            if embed is not None:
                found=True
                if opts.debug:
                    print(f'[smZNodes] using embedding:{embedding_name}')
                if embed.device != devices.device:
                    embed = embed.to(device=devices.device)
                if embedding_sname not in embeddings:
                    embeddings[embedding_sname] = {}
                embeddings[embedding_sname][clip_key] = embed
        if not found:
            print(f"warning, embedding:{embedding_name} does not exist, ignoring")
    # comfyui trims non-existent embedding_names while a1111 doesn't.
    # here we get group 2,5,6. group 2 minus its file extension.
    out = emb_re.sub(lambda m: (m.group(2) or '').removesuffix(m.group(4) or (m.group(3) or '')) + (m.group(5) or '') + (m.group(6) or ''), text_)
    for name, data in embeddings.items():
        emb = Embedding(data, name)
        shape = sum([v.shape[-1] for v in data.values()])
        vectors = max([v.shape[0] for v in data.values()])
        emb.shape = shape
        emb.vectors = vectors
        self.hijack.embedding_db.register_embedding(emb, self)
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
    fn = prompt_parser.reconstruct_multicond_batch if type(schedules).__name__ == "MulticondLearnedConditioning" else prompt_parser.reconstruct_cond_batch
    conds_list = [[(0, 1.0)]]
    cond = fn(schedules, step)
    if type(cond) is tuple:
        conds_list, cond = cond
    return (conds_list, cond)


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs, steps=0, current_step=0, multi=False):
        schedules = None
        texts = token_weight_pairs
        conds_list = [[(0, 1.0)]]
        from .modules.sd_hijack import model_hijack
        try:
            model_hijack.hijack(self)
            if isinstance(token_weight_pairs, list) and isinstance(token_weight_pairs[0], str):
                if multi:   schedules = prompt_parser.get_multicond_learned_conditioning(model_hijack.cond_stage_model, texts, steps, None, opts.use_old_scheduling)
                else:       schedules = prompt_parser.get_learned_conditioning(model_hijack.cond_stage_model, texts, steps, None, opts.use_old_scheduling)
                conds_list, cond = reconstruct_schedules(schedules, current_step)
                pooled = cond.pooled
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
                    return (zcond, first_pooled)
                # non-sdxl will be something like: {"l": [[]]}
                if isinstance(token_weight_pairs, dict):
                    token_weight_pairs = next(iter(token_weight_pairs.values()))
                cond, pooled = encode_toks(token_weight_pairs)
                cond.pooled = pooled
        finally:
                model_hijack.undo_hijack(model_hijack.cond_stage_model)
        device = model_management.intermediate_device() if hasattr(model_management, 'intermediate_device') else torch.device('cpu')
        return (cond.to(device), 
                {'pooled_output': pooled.to(device), 'schedules': schedules, 'conds_list': conds_list})

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

        if 'schedules' in g_pooled and 'schedules' in l_pooled:
            g_pooled = {"g": g_pooled, "l": l_pooled}

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
        # if hasattr(g_pooled, 'schedules'):
        #     g_pooled.schedules = {"g": g_pooled.schedules}
        return (g_out, g_pooled)

def is_prompt_editing(schedules):
    if schedules == None: return False
    if not isinstance(schedules, dict):
        schedules = {'g': schedules}
    ret = False
    for k,v in schedules.items():
        if type(v) is dict and 'schedules' in v:
            v=v['schedules']
        if type(v) == list:
            for vb in v:
                if len(vb) != 1: ret = True
        else:
            if v:
                for vb in v.batch:
                    for cs in vb:
                        if len(cs.schedules) != 1: ret = True
    return ret

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

def _find_outer_instance(target:str, target_type=None, callback=None):
    import inspect
    frame = inspect.currentframe()
    i = 0
    while frame and i < 10:
        if target in frame.f_locals:
            if callback is not None:
                return callback(frame)
            else:
                found = frame.f_locals[target]
                if isinstance(found, target_type):
                    return found
        frame = frame.f_back
        i += 1
    return None

if hasattr(comfy.model_patcher, 'ModelPatcher'):
    from comfy.model_patcher import ModelPatcher
else:
    ModelPatcher = object()
def prepare_noise(latent_image, seed, noise_inds=None, device='cpu'):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    opts = None
    model = _find_outer_instance('model', ModelPatcher)
    if (model is not None and (opts:=model.model_options.get('smZ_opts', None)) is None) or opts is None:
        import comfy.samplers
        guider = _find_outer_instance('guider', comfy.samplers.CFGGuider)
        model = getattr(guider, 'model_patcher', None)
    if (model is not None and (opts:=model.model_options.get('smZ_opts', None)) is None) or opts is None:
        import comfy.sample
        return comfy.sample.prepare_noise_orig(latent_image, seed, noise_inds)

    if opts.randn_source == 'gpu':
        device = model_management.get_torch_device()

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
    debug=opts.debug # get global opts' debug
    if (opts_new := clip_clone.patcher.model_options.get('smZ_opts', None)) is not None:
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
    opts.CLIP_stop_at_last_layers = abs(clip.layer_idx or 1)
    is_sdxl = "SDXL" in type(clip.cond_stage_model).__name__
        
    parser_map = {
        "full": "Full parser",
        "compel": "Compel parser",
        "A1111": "A1111 parser",
        "fixed attention": "Fixed attention",
        "comfy++": "Comfy++ parser",
    }
    opts.prompt_attention = parser_map.get(parser, "Comfy parser")

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
        out = out[0] # destructure tutple
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
        gen_id = lambda : binascii.hexlify(os.urandom(1024))[64:72]
        schedules = pooled['schedules'] if 'schedules' in pooled else pooled
        
        if opts.debug: print('[smZNodes] using steps', steps)
        _is_prompt_editing = is_prompt_editing(schedules)
        if not _is_prompt_editing: steps = 1
        conds=[]
        pooled_outputs = []

        if parser == 'comfy++':
            steps=0
            conds = [[cond]]
            if 'pooled_output' not in pooled:
                pooled = pooled.get('g', pooled.get('h', pooled.get('l', pooled)))
            pooled_outputs = [[pooled['pooled_output'] if 'pooled_output' in pooled  else pooled]]
            conds_list = pooled['conds_list'] if 'conds_list' in pooled else [[(0, 1.0)]]

        for x in range(0,steps):
            if schedules is None: continue
            if 'schedules' in pooled:
                conds_list, cond = reconstruct_schedules(schedules, x)
            else:
                conds_list, g_out = reconstruct_schedules(schedules['g']['schedules'], x)
                _conds_list, l_out = reconstruct_schedules(schedules['l']['schedules'], x)
                g_out, l_out = expand(g_out, l_out)
                l_out, g_out = expand(l_out, g_out)
                cond = torch.cat([l_out, g_out], dim=-1)
                cond.pooled = g_out.pooled

            if conds == []:
                conds=[[] for _ in range(cond.shape[0])]
                pooled_outputs=[[] for _ in range(cond.shape[0])]

            for ix, icond in enumerate(cond.chunk(cond.shape[0])):
                conds[ix].append(icond)
            for ix, ipo in enumerate(cond.pooled.chunk(cond.pooled.shape[0])):
                pooled_outputs[ix].append(ipo)

        # if all the same, only take the first cond
        for ix in range(len(conds)):
            if schedules is None: continue
            if all((conds[ix][0] == icond).all().item() for icond in conds[ix]):
                conds[ix] = [conds[ix][0]]
            # for ix in range(len(pooled_outputs)):
            if all((pooled_outputs[ix][0] == ipooled).all().item() for ipooled in pooled_outputs[ix]):
                pooled_outputs[ix] = [pooled_outputs[ix][0]]

        out=[]
        for ix, icl in enumerate(conds):
            id = gen_id()
            current_conds_list=[[deepcopy(conds_list[0][ix])]]
            for ixx, icond in enumerate(icl):
                _pooled = {"pooled_output": pooled_outputs[ix][ixx], "from_smZ": True, "smZid": id, "conds_list": current_conds_list, **sdxl_params}
                out.append([icond, _pooled])
    
    out[0][1]['smZ_opts'] = deepcopy(opts)

    return (out,)

# ========================================================================

from server import PromptServer
def prompt_handler(json_data):
    data=json_data['prompt']
    steps_validator = lambda x: isinstance(x, (int, float, str))
    text_validator = lambda x: isinstance(x, str)
    def find_nearest_ksampler(clip_id):
        """Find the nearest KSampler node that references the given CLIPTextEncode id."""
        nonlocal data, steps_validator
        for ksampler_id, node in data.items():
            if "Sampler" in node["class_type"] or "sampler" in node["class_type"]:
                # Check if this KSampler node directly or indirectly references the given CLIPTextEncode node
                if check_link_to_clip(ksampler_id, clip_id):
                    return get_val(data, ksampler_id, steps_validator, 'steps')
        return None

    def get_val(graph, node_id, validator, val):
        node = graph.get(str(node_id), {})
        if val == 'steps':
            steps_input_value = node.get("inputs", {}).get("steps", None)
            if steps_input_value is None:
                steps_input_value = node.get("inputs", {}).get("sigmas", None)
        else:
            steps_input_value = node.get("inputs", {}).get(val, None)

        while(True):
            # Base case: it's a direct value
            if not isinstance(steps_input_value, list) and validator(steps_input_value):
                if val == 'steps':
                    return min(max(1, int(steps_input_value)), 10000)
                else:
                    return steps_input_value
            # Loop case: it's a reference to another node
            elif isinstance(steps_input_value, list):
                ref_node_id, ref_input_index = steps_input_value
                ref_node = graph.get(str(ref_node_id), {})
                steps_input_value = ref_node.get("inputs", {}).get(val, None)
                if steps_input_value is None:
                    keys = list(ref_node.get("inputs", {}).keys())
                    ref_input_key = keys[ref_input_index % len(keys)]
                    steps_input_value = ref_node.get("inputs", {}).get(ref_input_key)
            else:
                return None

    def check_link_to_clip(node_id, clip_id, visited=None):
        """Check if a given node links directly or indirectly to a CLIPTextEncode node."""
        nonlocal data
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


    # Update each CLIPTextEncode node's steps with the steps from its nearest referencing KSampler node
    for clip_id, node in data.items():
        if node["class_type"] == "smZ CLIPTextEncode":
            check_str = prompt_editing = False
            if check_str:
                if (fast_search:=True):
                    with_SDXL = get_val(data, clip_id, lambda x: isinstance(x, (bool, int, float)), 'with_SDXL')
                    if with_SDXL:
                        ls = is_prompt_editing_str(get_val(data, clip_id, text_validator, 'text_l'))
                        gs = is_prompt_editing_str(get_val(data, clip_id, text_validator, 'text_g'))
                        prompt_editing = ls or gs
                    else:
                        text  = get_val(data, clip_id, text_validator, 'text')
                        prompt_editing = is_prompt_editing_str(text)
                else:
                    text = get_val(data, clip_id, text_validator, 'text')
                    prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules([text], steps, None, False)
                    prompt_editing = sum([len(ps) for ps in prompt_schedules]) != 1
            if check_str and not prompt_editing: continue
            steps = find_nearest_ksampler(clip_id)
            if steps is not None:
                node["inputs"]["smZ_steps"] = steps
                if opts.debug:
                    print(f'[smZNodes] id: {clip_id} | steps: {steps}')
    return json_data

def is_prompt_editing_str(t: str):
    """
    Determine if a string includes prompt editing.
    This won't cover every case, but it does the job for most.
    """
    if t is None: return True
    if (openb:=t.find('[')) != -1:
        if (colon:=t.find(':', openb)) != -1 and t.find(']', colon) != -1:
                return True
        elif (pipe:=t.find('|', openb)) != -1 and t.find(']', pipe) != -1:
                return True
    return False

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
def get_cond(c, current_step, reverse=False):
    """Group by smZ conds that may do prompt-editing / regular conds / comfy conds."""
    if not reverse: _cond = []
    else: _all = []
    fn2=lambda x : getp(x).get("smZid", None)
    prompt_editing = False
    for key, group in itertools.groupby(c, fn2):
        lsg=list(group)
        if key is not None:
            lsg_len = len(lsg)
            i = current_step if current_step < lsg_len else -1
            if lsg_len != 1: prompt_editing = True
            if not reverse: _cond.append(lsg[i])
            else: _all.append(lsg)
        else:
            if not reverse: _cond.extend(lsg)
            else:
                lsg.reverse()
                _all.append(lsg)
    
    if reverse:
        ls=_all
        ls.reverse()
        result=[]
        for d in ls:
            if isinstance(d, list):
                result.extend(d)
            else:
                result.append(d)
        del ls,_all
        return (result, prompt_editing)
    return (_cond, prompt_editing)

try:
    CFGGuiderOrig = comfy.samplers.CFGGuider_orig_smz = comfy.samplers.CFGGuider
except Exception:
    CFGGuiderOrig = comfy.samplers.CFGGuider_orig_smz = comfy.samplers.CFGNoisePredictor
class CFGGuider(CFGGuiderOrig):
    def __init__(self, model):
        self.conds = {}
        super().__init__(model)
        self.step = 0
        self.c_adm = None
        self.init_cond = None
        self.init_uncond = None
        self.is_prompt_editing_c = True
        self.is_prompt_editing_u = True
        self.use_CFGDenoiser = None
        self.opts = None
        self.sampler = None
        self.steps_multiplier = 1

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def apply_model(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, *args, **kwargs):
        if not hasattr(self, 'inner_model2'):
            self.inner_model2 = CFGDenoiser(self.inner_model.apply_model)
        x=kwargs['x'] if 'x' in kwargs else args[0]
        timestep=kwargs['timestep'] if 'timestep' in kwargs else args[1]
        cond=kwargs['cond'] if 'cond' in kwargs else (positive if (positive:=self.conds.get("positive", None)) is not None else args[2])
        uncond=kwargs['uncond'] if 'uncond' in kwargs else (negative if (negative:=self.conds.get("negative", None)) is not None else args[3])
        cond_scale=self.cfg if hasattr(self, 'cfg') else (kwargs['cond_scale'] if 'cond_scale' in kwargs else args[4])
        model_options=kwargs['model_options'] if 'model_options' in kwargs else {}

        # reverse doesn't work for some reason???
        # if self.init_cond is None:
        #     if len(cond) != 1 and any(['smZid' in ic for ic in cond]):
        #         self.init_cond = get_cond(cond, self.step, reverse=True)[0]
        #     else:
        #         self.init_cond = cond
        # cond = self.init_cond

        # if self.init_uncond is None:
        #     if len(uncond) != 1 and any(['smZid' in ic for ic in uncond]):
        #         self.init_uncond = get_cond(uncond, self.step, reverse=True)[0]
        #     else:
        #         self.init_uncond = uncond
        # uncond = self.init_uncond

        if self.is_prompt_editing_c:
            cc, ccp=get_cond(cond, self.step // self.steps_multiplier)
            self.is_prompt_editing_c=ccp
        else: cc = cond

        if self.is_prompt_editing_u:
            uu, uup=get_cond(uncond, self.step // self.steps_multiplier)
            self.is_prompt_editing_u=uup
        else: uu = uncond
        self.step += 1

        if 'transformer_options' not in model_options:
            model_options['transformer_options'] = {}

        if (any([getp(p).get('from_smZ', False) for p in cc]) or
            any([getp(p).get('from_smZ', False) for p in uu])):
            model_options['transformer_options']['from_smZ'] = True

        if not model_options['transformer_options'].get('from_smZ', False):
            if self.conds:
                out = super().predict_noise(*args, **kwargs)
            else:
                out = super().apply_model(*args, **kwargs)
            return out

        if self.is_prompt_editing_c:
            if 'cond' in kwargs: kwargs['cond'] = cc
            else: 
                if self.conds:
                    cbackup = self.conds['positive']
                    self.conds['positive'] = cc
                else:
                    args[2]=cc
        if self.is_prompt_editing_u:
            if 'uncond' in kwargs: kwargs['uncond'] = uu
            else:
                if self.conds:
                    ubackup = self.conds['negative']
                    self.conds['negative'] = uu
                else:
                    args[3]=uu

        if (self.is_prompt_editing_c or self.is_prompt_editing_u) and not self.sampler:
            def get_sampler(frame):
                return frame.f_code.co_name
            self.sampler = _find_outer_instance('extra_args', callback=get_sampler) or 'unknown'
            second_order_samplers = ["dpmpp_2s", "dpmpp_sde", "dpm_2", "heun"]
            # heunpp2 can be first, second, or third order
            third_order_samplers = ["heunpp2"]
            self.steps_multiplier = 2 if any(map(self.sampler.__contains__, second_order_samplers)) else self.steps_multiplier
            self.steps_multiplier = 3 if any(map(self.sampler.__contains__, third_order_samplers)) else self.steps_multiplier

        if self.use_CFGDenoiser is None:
            multi_cc = (any([getp(p)['smZ_opts'].multi_conditioning if 'smZ_opts' in getp(p) else False for p in cc]) and len(cc) > 1)
            multi_uu = (any([getp(p)['smZ_opts'].multi_conditioning if 'smZ_opts' in getp(p) else False for p in uu]) and len(uu) > 1)
            _opts = model_options.get('smZ_opts', None)
            if _opts is not None:
                self.inner_model2.opts = _opts
            self.use_CFGDenoiser = getattr(_opts, 'use_CFGDenoiser', multi_cc or multi_uu)

        # extends a conds_list to the number of latent images
        if self.use_CFGDenoiser and not hasattr(self.inner_model2, 'conds_list'):
            conds_list = []
            for ccp in cc:
                cpl = ccp['conds_list'] if 'conds_list' in ccp else [[(0, 1.0)]]
                conds_list.extend(cpl[0])
            conds_list=[conds_list]
            ix=-1
            cl = conds_list * len(x)
            conds_list=[list(((ix:=ix+1), zl[1]) for zl in cll) for cll in cl]
            self.inner_model2.conds_list = conds_list

        # to_comfy = not opts.debug
        to_comfy = True
        if self.use_CFGDenoiser and not to_comfy:
            _cc = torch.cat([c['model_conds']['c_crossattn'].cond for c in cc])
            _uu = torch.cat([c['model_conds']['c_crossattn'].cond for c in uu])

        # reverse conds here because comfyui reverses them later
        if len(cc) != 1 and any(['smZid' in ic for ic in cond]):
            cc = list(reversed(cc))
            if 'cond' in kwargs: kwargs['cond'] = cc
            else:
                if self.conds:
                    self.conds['positive'] = cc
                else:
                    args[2]=cc
        if len(uu) != 1 and any(['smZid' in ic for ic in uncond]):
            uu = list(reversed(uu))
            if 'uncond' in kwargs: kwargs['uncond'] = uu
            else:
                if self.conds:
                    self.conds['negative'] = uu
                else:
                    args[3]=uu
        
        if not self.use_CFGDenoiser:
            kwargs['model_options'] = model_options
            if self.conds:
                out = super().predict_noise(*args, **kwargs)
            else:
                out = super().apply_model(*args, **kwargs)
            if self.conds:
                if self.is_prompt_editing_c:
                    self.conds['positive'] = cbackup
                if self.is_prompt_editing_u:
                    self.conds['negative'] = ubackup
        else:
            self.inner_model2.x_in = x
            self.inner_model2.sigma = timestep
            self.inner_model2.cond_scale = cond_scale
            self.inner_model2.image_cond = image_cond = None
            if 'x' in kwargs: kwargs['x'].conds_list = self.inner_model2.conds_list
            else: args[0].conds_list = self.inner_model2.conds_list
            if not hasattr(self.inner_model2, 's_min_uncond'):
                self.inner_model2.s_min_uncond = getattr(model_options.get('smZ_opts', None), 's_min_uncond', 0)
            if 'model_function_wrapper' in model_options:
                model_options['model_function_wrapper_orig'] = model_options.pop('model_function_wrapper')
            if to_comfy:
                model_options["model_function_wrapper"] = self.inner_model2.forward_
            else:
                if 'sigmas' not in model_options['transformer_options']:
                    model_options['transformer_options']['sigmas'] = timestep
            self.inner_model2.model_options = kwargs['model_options'] = model_options
            if not hasattr(self.inner_model2, 'skip_uncond'):
                self.inner_model2.skip_uncond = math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False
            if to_comfy:
                if self.conds:
                    kwargs['cond'] = self.conds['positive']
                    kwargs['uncond'] = self.conds['negative']
                    kwargs['cond_scale'] = cond_scale
                    out = sampling_function(self.inner_model, *args, **kwargs)
                    if self.is_prompt_editing_c:
                        self.conds['positive'] = cbackup
                    if self.is_prompt_editing_u:
                        self.conds['negative'] = ubackup
                else:
                    out = sampling_function(self.inner_model, *args, **kwargs)
            else:
                out = self.inner_model2(x, timestep, cond=_cc, uncond=_uu, cond_scale=cond_scale, s_min_uncond=self.inner_model2.s_min_uncond, image_cond=image_cond)
        return out


def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        else:
            uncond_ = uncond

        cfg_result = None
        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options, cond_scale)
        if hasattr(x, 'conds_list'): cfg_result = cond_pred

        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        else:
            if cfg_result is None:
                cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result

if hasattr(comfy.samplers, 'get_area_and_mult'):
    from comfy.samplers import get_area_and_mult, can_concat_cond, cond_cat
def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options, cond_scale_in):
    conds = []
    a1111 = hasattr(x_in, 'conds_list')

    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c['control'] = control if 'tiled_diffusion' in model_options else control.get_control(input_x, timestep_, c, len(cond_or_uncond))

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                if a1111:
                    out_cond_ = torch.zeros_like(x_in)
                    out_cond_[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                    conds.append(out_cond_)
                else:
                    out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult
    if not a1111:
        out_cond /= out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    if a1111:
        conds_len = len(conds)
        if conds_len != 0:
            lenc = max(conds_len,1.0)
            cond_scale = 1.0/lenc * (1.0 if "sampler_cfg_function" in model_options else cond_scale_in)
            conds_list = x_in.conds_list
            if (inner_conds_list_len:=len(conds_list[0])) < conds_len:
                conds_list = [[(ix, 1.0 if ix > inner_conds_list_len-1 else conds_list[0][ix][1]) for ix in range(conds_len)]]
            out_cond = out_uncond.clone()
            for cond, (_, weight) in zip(conds, conds_list[0]):
                out_cond += (cond / (out_count / lenc) - out_uncond) * weight * cond_scale

    del out_count
    return out_cond, out_uncond

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
# Hijack sampling

payload = [{
    "target_line": 'extra_args["denoise_mask"] = denoise_mask',
    "code_to_insert": """
            if (any([_p[1].get('from_smZ', False) for _p in positive]) or 
                any([_p[1].get('from_smZ', False) for _p in negative])):
                from ComfyUI_smZNodes.modules.shared import opts as smZ_opts
                if not smZ_opts.sgm_noise_multiplier: max_denoise = False
"""
},
{
    "target_line": 'positive = positive[:]',
    "code_to_insert": """
        if hasattr(self, 'model_denoise'): self.model_denoise.step = start_step if start_step != None else 0
"""
},
]

def hook_for_settings_node_and_sampling():
    if not hasattr(comfy.samplers, 'Sampler'):
        print(f"[smZNodes]: Your ComfyUI version is outdated. Please update to the latest version.")
        comfy.samplers.KSampler.sample = inject_code(comfy.samplers.KSampler.sample, payload)
    else:
        _KSampler_sample = comfy.samplers.KSampler.sample
        _Sampler = comfy.samplers.Sampler
        _max_denoise = comfy.samplers.Sampler.max_denoise
        _sample = comfy.samplers.sample

        def get_value_from_args(args, kwargs, key_to_lookup, fn, idx=None):
            value = None
            if key_to_lookup in kwargs:
                value = kwargs[key_to_lookup]
            else:
                try:
                    # Get its position in the formal parameters list and retrieve from args
                    arg_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
                    index = arg_names.index(key_to_lookup)
                    value = args[index] if index < len(args) else None
                except Exception as err:
                    if idx is not None and idx < len(args):
                        value = args[idx]
            return value

        def KSampler_sample(*args, **kwargs):
            start_step = get_value_from_args(args, kwargs, 'start_step', _KSampler_sample)
            if isinstance(start_step, int):
                args[0].model.start_step = start_step
            return _KSampler_sample(*args, **kwargs)

        def sample(*args, **kwargs):
            model = get_value_from_args(args, kwargs, 'model', _sample, 0)
            # positive = get_value_from_args(args, kwargs, 'positive', _sample, 2)
            # negative = get_value_from_args(args, kwargs, 'negative', _sample, 3)
            sampler = get_value_from_args(args, kwargs, 'sampler', _sample, 6)
            model_options = get_value_from_args(args, kwargs, 'model_options', _sample, 8)
            start_step = getattr(model, 'start_step', None)
            if 'smZ_opts' in model_options:
                model_options['smZ_opts'].start_step = start_step
                opts = model_options['smZ_opts']
                if hasattr(sampler, 'sampler_function'):
                    if not hasattr(sampler, 'sampler_function_orig'):
                        sampler.sampler_function_orig = sampler.sampler_function
                    sampler_function_sig_params = inspect.signature(sampler.sampler_function).parameters
                    params = {x: getattr(opts, x)  for x in ['eta', 's_churn', 's_tmin', 's_tmax', 's_noise'] if x in sampler_function_sig_params}
                    sampler.sampler_function = lambda *a, **kw: sampler.sampler_function_orig(*a, **{**kw, **params})
            # Add model_options to CFGGuider
            if hasattr(model, 'model_options') and model.model_options is dict:
                model.model_options.update(model_options)
            else:
                model.model_options = model_options
            return _sample(*args, **kwargs)

        class Sampler(_Sampler):
            def max_denoise(self, model_wrap: CFGGuider, sigmas):
                base_model = model_wrap.inner_model
                res = _max_denoise(self, model_wrap, sigmas)
                model_options = getattr(model_wrap, 'model_options', None)
                if model_options is None:
                    model_options = base_model.model_options
                if model_options is not None:
                    if 'smZ_opts' in model_options:
                        opts = model_options['smZ_opts']
                        if getattr(opts, 'start_step', None) is not None:
                            model_wrap.step = opts.start_step
                            opts.start_step = None
                        if not opts.sgm_noise_multiplier:
                            res = False
                return res

        comfy.samplers.Sampler.max_denoise = Sampler.max_denoise
        comfy.samplers.KSampler.sample = KSampler_sample
        comfy.samplers.sample = sample
    if hasattr(comfy.samplers, 'CFGGuider'):
        comfy.samplers.CFGGuider = CFGGuider
    elif hasattr(comfy.samplers, 'CFGNoisePredictor'):
        comfy.samplers.CFGNoisePredictor = CFGGuider

def hook_for_rng_orig():
    if not hasattr(comfy.sample, 'prepare_noise_orig'):
        comfy.sample.prepare_noise_orig = comfy.sample.prepare_noise

def hook_for_dtype_unet():
    if hasattr(comfy.model_management, 'unet_dtype'):
        if not hasattr(comfy.model_management, 'unet_dtype_orig'):
            comfy.model_management.unet_dtype_orig = comfy.model_management.unet_dtype
        from .modules import devices
        def unet_dtype(device=None, model_params=0, *args, **kwargs):
            dtype = comfy.model_management.unet_dtype_orig(device=device, model_params=model_params, *args, **kwargs)
            if model_params != 0:
                devices.dtype_unet = dtype
            return dtype
        comfy.model_management.unet_dtype = unet_dtype

def try_hook(fn):
    try:
        fn()
    except Exception as e:
        print("\033[92m[smZNodes] \033[0;33mWARNING:\033[0m", e)

def register_hooks():
    hooks = [
        hook_for_settings_node_and_sampling,
        hook_for_rng_orig,
        hook_for_dtype_unet,
    ]
    for hook in hooks:
        try_hook(hook)

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
        except ValueError as e: ...

def add_custom_samplers():
    samplers = [
        add_sample_dpmpp_2m_alt,
    ]
    for add_sampler in samplers:
        add_sampler()