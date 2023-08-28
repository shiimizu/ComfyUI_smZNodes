import comfy
import torch
from comfy.sd import CLIP
from typing import List, Tuple
from types import MethodType
from functools import partial
from .modules import prompt_parser, shared, devices
from .modules.shared import opts
from .modules.sd_samplers_kdiffusion import CFGDenoiser
from .modules.sd_hijack_clip import FrozenCLIPEmbedderForSDXLWithCustomWords, FrozenCLIPEmbedderWithCustomWordsBase
from .modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedder2WithCustomWords
from .modules.textual_inversion.textual_inversion import Embedding
import comfy.sdxl_clip
import comfy.sd1_clip
import comfy.sample
from comfy.sd1_clip import SD1Tokenizer, unescape_important, escape_important, token_weights, expand_directory_list
from comfy.sdxl_clip import SDXLClipGTokenizer
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
from nodes import CLIPTextEncode
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from comfy import model_base, model_management
from comfy.samplers import KSampler, CompVisVDenoiser, KSamplerX0Inpaint
from comfy.k_diffusion.external import CompVisDenoiser
from types import MethodType
import nodes
import inspect
from textwrap import dedent
import functools
import tempfile
import importlib
import sys
import os
import re

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

class PopulateVars():
    def populate_self_variables(self, from_):
        super_attrs = vars(from_)
        self_attrs = vars(self)
        self_attrs.update(super_attrs)

class FrozenOpenCLIPEmbedder2WithCustomWordsCustom(FrozenOpenCLIPEmbedder2WithCustomWords, SDXLClipGTokenizer, PopulateVars):
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
        return self.encode_with_transformers_comfy(tokens, return_pooled)

    def encode_token_weights(self, tokens):
        pass

    def encode_with_transformers_comfy(self, tokens: List[List[int]], return_pooled=False) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This function is different from `clip.cond_stage_model.encode_token_weights()`
        in that the tokens are `List[List[int]]`, not including the weights.

        Originally from `sd1_clip.py`: `encode()` -> `forward()`
        '''
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        z, pooled = self.wrapped(tokens)
        if z.device != devices.device:
            z = z.to(device=devices.device)
        # if z.dtype != devices.dtype:
        #     z = z.to(dtype=devices.dtype)
        # if pooled.dtype != devices.dtype:
        #     pooled = pooled.to(dtype=devices.dtype)
        z.pooled = pooled
        return (z, pooled) if return_pooled else z

    def tokenize(self, texts):
        # assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'
        tokenized = [self.tokenizer(text)["input_ids"][1:-1] for text in texts]
        return tokenized


class FrozenCLIPEmbedderWithCustomWordsCustom(FrozenCLIPEmbedderForSDXLWithCustomWords, PopulateVars):
    '''
    Custom class that also inherits a tokenizer to have the `_try_get_embedding()` method.
    '''
    def __init__(self, wrapped: comfy.sd1_clip.SD1ClipModel, hijack):
        self.populate_self_variables(wrapped.tokenizer_parent) # SD1Tokenizer
        # self.embedding_identifier_tokenized = wrapped.tokenizer([self.embedding_identifier])["input_ids"][0][1:-1]
        super().__init__(wrapped, hijack)

    def encode(self, tokens):
        return self.encode_with_transformers(tokens, True)

    def encode_with_transformers(self, tokens, return_pooled=False):
        return self.encode_with_transformers_comfy(tokens, return_pooled)

    def encode_token_weights(self, tokens):
        pass

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

    def encode_with_transformers__(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        elif self.wrapped.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]
            if self.wrapped.layer_norm_hidden_state:
                z = self.wrapped.transformer.text_model.final_layer_norm(z)

        pooled_output = outputs.pooler_output
        if self.wrapped.text_projection is not None:
            pooled_output = pooled_output.to(self.wrapped.text_projection.device) @ self.wrapped.text_projection
        if z.device != devices.device:
            z = z.to(device=devices.device)
        # z=z.float()
        # z.pooled = pooled_output.float()
        z.pooled = pooled_output
        return z

    def tokenize_line(self, line):
        line = parse_and_register_embeddings(self, line)
        return super().tokenize_line(line)

    def tokenize(self, texts):
        tokenized = [self.tokenizer(text)["input_ids"][1:-1] for text in texts]
        return tokenized

def tokenize_with_weights_custom(self:SD1Tokenizer, text:str, return_word_ids=False):
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
    embs_str = '|'.join(embs)
    embs_str = embs_str + '|' if embs_str else ''
    emb_re = r"(embedding:)?(?:({}[\w\.\-\!\$\/\\]+(\.safetensors|\.pt|\.bin)|(?(1)[\w\.\-\!\$\/\\]+|(?!)))(\.safetensors|\.pt|\.bin)?)(?::(\d+\.?\d*|\d*\.\d+))?".format(embs_str)
    emb_re = re.compile(emb_re, flags=re.MULTILINE | re.UNICODE | re.IGNORECASE)

    #tokenize words
    tokens = []
    for weighted_segment, weight in parsed_weights:
        to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            matches = emb_re.finditer(word)
            last_end = 0
            for _, match in enumerate(matches, start=1):
                start=match.start()
                end=match.end()
                nw=word[last_end:start]
                if nw:
                    tokens.append([(t, weight) for t in self.tokenizer(nw)["input_ids"][1:-1]])
                if (embedding_name := match.group(2)) is not None:
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        # print(f'using embedding:{embedding_name}')
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                last_end = end
            nw = word[last_end:]
            if nw:
                tokens.append([(t, weight) for t in self.tokenizer(nw)["input_ids"][1:-1]])

    #reshape token array to CLIP input size
    batched_tokens = []
    batch = [(self.start_token, 1.0, 0)]
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
                    batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                #start new batch
                batch = [(self.start_token, 1.0, 0)]
                batched_tokens.append(batch)
            else:
                batch.extend([(t,w,i+1) for t,w in t_group])
                t_group = []

    #fill last batch
    batch.extend([(self.end_token, 1.0, 0)] + [(pad_token, 1.0, 0)] * (self.max_length - len(batch) - 1))

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
    return embs

def parse_and_register_embeddings(self: FrozenCLIPEmbedderWithCustomWordsCustom|FrozenOpenCLIPEmbedder2WithCustomWordsCustom, text: str, return_word_ids=False):
    from  builtins import any as b_any
    def inverse_substring(s, start, end, offset = 0):
        return s[:start-offset] + s[end-offset:]

    embs = get_valid_embeddings(self.wrapped.tokenizer_parent.embedding_directory)
    embs_str = '|'.join(embs)
    emb_re = r"(embedding:)?(?:({}[\w\.\-\!\$\/\\]+(\.safetensors|\.pt|\.bin)|(?(1)[\w\.\-\!\$\/\\]+|(?!)))(\.safetensors|\.pt|\.bin)?)(?::(\d+\.?\d*|\d*\.\d+))?".format(embs_str + '|' if embs_str else '')
    emb_re = re.compile(emb_re, flags=re.MULTILINE | re.UNICODE | re.IGNORECASE)
    matches = emb_re.finditer(text)
    offset = 0
    for matchNum, match in enumerate(matches, start=1):
        found=False
        if (embedding_name := match.group(2)):
            embed, _ = self.wrapped.tokenizer_parent._try_get_embedding(embedding_name)
            if embed is not None:
                found=True
                # print(f'using embedding:{embedding_name}')
                if embed.device != devices.device:
                    embed = embed.to(device=devices.device)
                self.hijack.embedding_db.register_embedding(Embedding(embed, embedding_name), self)
        if not found:
            print(f"warning, embedding:{embedding_name} does not exist, ignoring")
            text = inverse_substring(text, match.start(), match.end(), offset)
            offset += len(embedding_name)
    return text.replace('embedding:', '')

def expand(t1, t2, empty_t=None, with_zeros=False):
    if t1.shape[1] < t2.shape[1]:
        if with_zeros:
            if empty_t == None: empty_t = shared.sd_model.cond_stage_model_empty_prompt
            num_repetitions = (t2.shape[1] - t1.shape[1]) // empty_t.shape[1]
            return torch.cat([t1, empty_t.repeat((t1.shape[0], num_repetitions, 1))], axis=1)
        else:
            num_repetitions = t2.shape[1] // t1.shape[1]
            return t1.repeat(1, num_repetitions, 1)
    else:
        return t1

def reconstruct_schedules(schedules, step):
    create_reconstruct_fn = lambda _cc: prompt_parser.reconstruct_multicond_batch if type(_cc).__name__ == "MulticondLearnedConditioning" else prompt_parser.reconstruct_cond_batch
    reconstruct_fn = create_reconstruct_fn(schedules)
    return reconstruct_fn(schedules, step)


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs, steps=0, current_step=0, multi=False):
        schedules = token_weight_pairs
        texts = token_weight_pairs
        if isinstance(token_weight_pairs, list) and isinstance(token_weight_pairs[0], str):
            from .modules.sd_hijack import model_hijack
            try:
                model_hijack.hijack(self)
                # from A1111's processing.py and sd_samplers_kdiffusion.py
                if multi:
                    schedules = prompt_parser.get_multicond_learned_conditioning(model_hijack.cond_stage_model, texts, steps)
                else:
                    schedules = prompt_parser.get_learned_conditioning(model_hijack.cond_stage_model, texts, steps)
                model_hijack.undo_hijack(model_hijack.cond_stage_model)
            except Exception as err:
                try:
                    model_hijack.undo_hijack(model_hijack.cond_stage_model)
                except Exception as err2:
                    pass
                raise err
        elif type(token_weight_pairs) == list and type(token_weight_pairs[0]) == list and type(token_weight_pairs[0][0]) == tuple:
            # comfy++
            from .modules.sd_hijack import model_hijack
            try:
                model_hijack.hijack(self)
                def encode_toks(_token_weight_pairs):
                    zs = []
                    first_pooled = None
                    for batch_chunk in _token_weight_pairs:
                        tokens_ = [x[0] for x in batch_chunk]
                        multipliers = [x[1] for x in batch_chunk]
                        z = model_hijack.cond_stage_model.process_tokens([tokens_], [multipliers])
                        if first_pooled == None:
                            first_pooled = z.pooled
                        zs.append(z)
                    zcond = torch.hstack(zs)
                    zcond.pooled = first_pooled
                    return zcond
                cond = encode_toks(token_weight_pairs) 
                model_hijack.undo_hijack(model_hijack.cond_stage_model)
            except Exception as err:
                try:
                    model_hijack.undo_hijack(model_hijack.cond_stage_model)
                except Exception as err2:
                    pass
                raise err
            conds_list = [[(0, 1.0)]]
            cond.pooled.conds_list = conds_list
            return (cond, cond.pooled)
        conds_list = [[(0, 1.0)]]
        cond = reconstruct_schedules(schedules, current_step)
        if type(cond) == tuple:
            conds_list, cond = cond
        cond.pooled.conds_list = conds_list
        return (cond, cond.pooled)

class SD1ClipModel(ClipTokenWeightEncoder, comfy.sd1_clip.SD1ClipModel, PopulateVars):
    def __init__(self):
        pass
    def wrap(self, clip_model, tokenizer):
        self.populate_self_variables(clip_model)
        self.tokenizer = tokenizer
        return self

class SDXLClipG(ClipTokenWeightEncoder, comfy.sdxl_clip.SDXLClipG, PopulateVars):
    def __init__(self):
        pass
    def wrap(self, clip_model, tokenizer):
        self.populate_self_variables(clip_model)
        self.tokenizer = tokenizer
        return self

class SDXLClipModel(comfy.sdxl_clip.SDXLClipModel, PopulateVars):
    def __init__(self):
        pass
    def wrap(self, clip_model, tokenizer: comfy.sdxl_clip.SDXLTokenizer):
        self.populate_self_variables(clip_model)
        self.clip_l_orig = self.clip_l
        self.clip_g_orig = self.clip_g
        self.clip_l_ = SD1ClipModel().wrap(self.clip_l, tokenizer.clip_l)
        self.clip_g_ = SDXLClipG().wrap(self.clip_g, tokenizer.clip_g)
        self.tokenizer = tokenizer
        return self

    def encode_token_weights(self, token_weight_pairs, steps=0, current_step=0, multi=False):
        self.clip_g = self.clip_g_
        self.clip_l = self.clip_l_
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g, steps, current_step, multi)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l, steps, current_step, multi)
        if hasattr(g_pooled, 'schedules') and hasattr(l_pooled, 'schedules'):
            g_pooled.schedules = {"g": g_pooled.schedules, "l": l_pooled.schedules}
        if opts.pad_with_repeats:
            # Both yield the same results
            # g_out = expand(g_out, l_out, shared.sd_model.cond_stage_model_empty_prompt_g, True)
            # l_out = expand(l_out, g_out, shared.sd_model.cond_stage_model_empty_prompt_l, True)
            g_out = expand(g_out, l_out)
            l_out = expand(l_out, g_out)
        self.clip_l = self.clip_l_orig
        self.clip_g = self.clip_g_orig
        return torch.cat([l_out, g_out], dim=-1), g_pooled

class SDXLRefinerClipModel(comfy.sdxl_clip.SDXLRefinerClipModel, PopulateVars):
    def __init__(self):
        pass
    def wrap(self, clip_model, tokenizer: comfy.sdxl_clip.SDXLTokenizer):
        self.populate_self_variables(clip_model)
        self.clip_g_orig = self.clip_g
        self.clip_g_ = SDXLClipG().wrap(self.clip_g, tokenizer.clip_g)
        self.tokenizer = tokenizer
        return self
    def encode_token_weights(self, token_weight_pairs, steps=0, current_step=0, multi=False):
        self.clip_g = self.clip_g_
        token_weight_pairs_g = token_weight_pairs["g"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g, steps, current_step, multi)
        self.clip_g = self.clip_g_orig
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

def _find_outer_instance(target, target_type):
    import inspect
    frame = inspect.currentframe()
    while frame:
        if target in frame.f_locals:
            found = frame.f_locals[target]
            if isinstance(found, target_type) and found != 1: # steps == 1
                return found
        frame = frame.f_back
    return None

class LazyCond:
    def __init__(self, cond = None):
        self.cond = cond
        self.steps = 1

    def __get__(self, instance, owner):
        return self._get()

    def _get(self):
        if (steps := _find_outer_instance('steps', int)) is None:
            steps = self.steps
        if self.steps != steps:
            self.steps = steps
            if self._is_prompt_editing():
                params = self.cond[0][1]['params']
                params['steps'] = steps
                params.pop('step', None)
                res = run(**params)[0]
                res[0][1]['params'] = {}
                res[0][1]['params'].update(params)
                self.cond = res
        return self.cond

    def _is_prompt_editing(self):
        params = self.cond[0][1]['params']
        text = params['text']
        text_g = params['text_g']
        text_l = params['text_l']
        with_SDXL = params['with_SDXL']
        is_SDXL = "SDXL" in type(params['clip'].cond_stage_model).__name__
        text_all = (text_g + ' ' + text_l) if with_SDXL and is_SDXL else text
        found = '[' in text_all and ']' in text_all
        return found or is_prompt_editing(getattr(self.cond[0][1]['pooled_output'], 'schedules', None))

    def __iter__(self):
        return self._get().__iter__()

def run(clip: comfy.sd.CLIP, text, parser, mean_normalization,
               multi_conditioning, use_old_emphasis_implementation,
               use_CFGDenoiser, with_SDXL, ascore, width, height, crop_w, 
               crop_h, target_width, target_height, text_g, text_l, steps=1, step=0, with_pooled=None):
    opts.prompt_mean_norm = mean_normalization
    opts.use_old_emphasis_implementation = use_old_emphasis_implementation
    opts.CLIP_stop_at_last_layers = abs(clip.layer_idx or 1)
    is_sdxl = "SDXL" in type(clip.cond_stage_model).__name__
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
        opts.use_CFGDenoiser = use_CFGDenoiser # unused

    sdxl_conds = {}
    if with_SDXL and is_sdxl:
        sdxl_conds = {
            "aesthetic_score": ascore, "width": width, "height": height,
            "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width,
            "target_height": target_height, "text_g": text_g, "text_l": text_l
        }
    pooled={}
    if not hasattr(SD1Tokenizer, 'tokenize_with_weights_orig'):
        SD1Tokenizer.tokenize_with_weights_orig = SD1Tokenizer.tokenize_with_weights
    if parser == "comfy":
        SD1Tokenizer.tokenize_with_weights = tokenize_with_weights_custom
        clip_model_type_name = type(clip.cond_stage_model).__name__ 
        if with_SDXL and is_sdxl:
            if clip_model_type_name== "SDXLClipModel":
                out = CLIPTextEncodeSDXL().encode(clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)
                out[0][0][1]['aesthetic_score'] = sdxl_conds['aesthetic_score']
            elif clip_model_type_name == "SDXLRefinerClipModel":
                out = CLIPTextEncodeSDXLRefiner().encode(clip, ascore, width, height, text)
                for item in ['aesthetic_score', 'width', 'height', 'text_g', 'text_l']:
                    sdxl_conds.pop(item)
                out[0][0][1].update(sdxl_conds)
        else:
            out = CLIPTextEncode().encode(clip, text)
        if hasattr(SD1Tokenizer, 'tokenize_with_weights_orig'):
            SD1Tokenizer.tokenize_with_weights = SD1Tokenizer.tokenize_with_weights_orig
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

        clip_clone = clip.clone()
        # clip_clone = clip
        def get_custom_cond_stage_model():
            nonlocal clip_clone
            from .smZNodes import SD1ClipModel, SDXLClipModel, SDXLRefinerClipModel
            clip_model_type_name = type(clip_clone.cond_stage_model).__name__
            tokenizer = clip_clone.tokenizer
            if clip_model_type_name== "SDXLClipModel":
                return SDXLClipModel().wrap(clip_clone.cond_stage_model, tokenizer)
            elif clip_model_type_name == "SDXLRefinerClipModel":
                return SDXLRefinerClipModel().wrap(clip_clone.cond_stage_model, tokenizer)
            else:
                return SD1ClipModel().wrap(clip_clone.cond_stage_model, tokenizer)

        tokens = texts
        if with_pooled != None:
            tokens = with_pooled['pooled_output'].schedules
        if parser == "comfy++":
            SD1Tokenizer.tokenize_with_weights = tokenize_with_weights_custom
            tokens = clip_clone.tokenize(text)
            if hasattr(SD1Tokenizer, 'tokenize_with_weights_orig'):
                SD1Tokenizer.tokenize_with_weights = SD1Tokenizer.tokenize_with_weights_orig
        cond = pooled = None

        # Because of prompt editing, we need the total number of steps
        # So this function will be called back at the sampling stage
        # Note that this means encoding will happen twice
        cond_stage_model_orig = clip_clone.cond_stage_model
        if with_pooled == None:
            clip_clone.cond_stage_model = get_custom_cond_stage_model()
        else:
            clip_clone.load_model_orig = clip_clone.load_model
            clip_clone.load_model = lambda x=None:x
            clip_clone.cond_stage_model = with_pooled['pooled_output'].cond_stage_model
        clip_clone.cond_stage_model.encode_token_weights = partial(clip_clone.cond_stage_model.encode_token_weights, steps=steps, current_step=step, multi=multi_conditioning)
        cond, pooled = clip_clone.encode_from_tokens(tokens, True)
        if with_pooled == None:
            pooled.cond_stage_model = clip_clone.cond_stage_model
        else:
            clip_clone.load_model = clip_clone.load_model_orig
        clip_clone.cond_stage_model = cond_stage_model_orig



        pooled = {"pooled_output": pooled, "from_smZ": True, "use_CFGDenoiser": use_CFGDenoiser, **sdxl_conds}
    return ([[cond, pooled if with_pooled == None else with_pooled]], )

# ========================================================================

class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ksampler = _find_outer_instance('self', comfy.samplers.KSampler)
        self.step = 0
        self.orig = comfy.samplers.CFGNoisePredictorOrig(model)
        self.inner_model = CFGDenoiser(model.apply_model)
        self.inner_model.num_timesteps = model.num_timesteps
        self.s_min_uncond = opts.s_min_uncond
        self.inner_model.device = self.ksampler.device if hasattr(self.ksampler, "device") else devices.device
        self.alphas_cumprod = model.alphas_cumprod
        self.c_adm = None
        self.init_cond = None
        self.init_uncond = None
        self.is_prompt_editing_u = False
        self.is_prompt_editing_c = False

    def apply_model(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, model_options={}, seed=None):
        if not (cond[0][1].get('from_smZ', False) and uncond[0][1].get('from_smZ', False)):
            return self.orig.apply_model(x, timestep, cond, uncond, cond_scale, cond_concat, model_options, seed)

        if self.init_cond is None:
            self.init_cond = cond
            self.init_uncond = uncond
            self.is_prompt_editing_c = is_prompt_editing(getattr(cond[0][1]['pooled_output'], 'schedules', None))
            self.is_prompt_editing_u = is_prompt_editing(getattr(uncond[0][1]['pooled_output'], 'schedules', None))
        if self.step != 0:
            cond = self.init_cond
            if self.is_prompt_editing_c:
                params = cond[0][1]['params']
                params['step'] = self.step
                cond = run(with_pooled=cond[0][1], **params)[0]

            uncond = self.init_uncond
            if self.is_prompt_editing_u:
                params = uncond[0][1]['params']
                params['step'] = self.step
                uncond = run(with_pooled=uncond[0][1], **params)[0]

            if (self.is_prompt_editing_c or self.is_prompt_editing_u) and (not torch.all(self.init_cond[0][0] == cond[0][0]).item() or not torch.all(self.init_uncond[0][0] == uncond[0][0]).item()):
                cond, uncond = process_conds_comfy(self.ksampler, cond, uncond)
            
        co = cond[0][0]
        unc = uncond[0][0]
        if cond[0][1].get('from_smZ', False) and "comfy" != cond[0][1].get('params',{}).get('parser', 'comfy'):
            co = expand(co, unc)
        if uncond[0][1].get('from_smZ', False) and "comfy" != uncond[0][1].get('params',{}).get('parser', 'comfy'):
            unc = expand(unc, co)


        if cond[0][1].get('use_CFGDenoiser', False) or uncond[0][1].get('use_CFGDenoiser', False):
            if cond[0][1].get("adm_encoded", None) != None and self.c_adm == None:
                self.c_adm = torch.cat([cond[0][1]['adm_encoded'], uncond[0][1]['adm_encoded']])
            x.c_adm = self.c_adm
            conds_list = cond[0][1]['pooled_output'].conds_list
            cond = (conds_list, co)
            image_cond = txt2img_image_conditioning(None, x)
            out = self.inner_model(x, timestep, cond=cond, uncond=unc, cond_scale=cond_scale, s_min_uncond=self.s_min_uncond, image_cond=image_cond)
        else:
            cond = [[co, cond[0][1]]]
            uncond = [[unc, uncond[0][1]]]
            out = self.orig.apply_model(x, timestep, cond, uncond, cond_scale, cond_concat, model_options, seed)
        self.step += 1
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

def process_conds_comfy(self, positive, negative):
        from comfy.sample import broadcast_cond
        from comfy.samplers import (resolve_cond_masks, calculate_start_end_timesteps,
                                            create_cond_with_same_area_if_none, pre_run_control,
                                            apply_empty_x_to_equal_area, encode_adm)
        noise = getattr(self.model_k, "noise", opts.noise)
        if not hasattr(self, "device"):
            self.device = devices.device
        device = self.device

        positive = broadcast_cond(positive, noise.shape[0], device)
        negative = broadcast_cond(negative, noise.shape[0], device)

        positive = positive[:]
        negative = negative[:]

        resolve_cond_masks(positive, noise.shape[2], noise.shape[3], self.device)
        resolve_cond_masks(negative, noise.shape[2], noise.shape[3], self.device)

        calculate_start_end_timesteps(self.model_wrap, negative)
        calculate_start_end_timesteps(self.model_wrap, positive)

        #make sure each cond area has an opposite one with the same area
        for c in positive:
            create_cond_with_same_area_if_none(negative, c)
        for c in negative:
            create_cond_with_same_area_if_none(positive, c)

        pre_run_control(self.model_wrap, negative + positive)

        apply_empty_x_to_equal_area(list(filter(lambda c: c[1].get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
        apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

        if self.model.is_adm():
            positive = encode_adm(self.model, positive, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "positive")
            negative = encode_adm(self.model, negative, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "negative")
        return (positive, negative)

# =======================================================================================

def set_model_k(self: KSampler):
    self.model_denoise = CFGNoisePredictor(self.model, self) # main change
    if ((getattr(self.model, "parameterization", "") == "v") or
        (getattr(self.model, "model_type", -1) == model_base.ModelType.V_PREDICTION)):
        self.model_wrap = CompVisVDenoiser(self.model_denoise, quantize=True)
        self.model_wrap.parameterization = getattr(self.model, "parameterization", "v")
    else:
        self.model_wrap = CompVisDenoiser(self.model_denoise, quantize=True)
        self.model_wrap.parameterization = getattr(self.model, "parameterization", "eps")
    self.model_k = KSamplerX0Inpaint(self.model_wrap)

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
        KSampler_orig = comfy.samplers.KSampler
        ret = None
        try:
            comfy.samplers.KSampler = SDKSampler
            ret = super(smz_SDKSampler, self).sample(*args, **kwargs)
            comfy.samplers.KSampler = KSampler_orig
        except Exception as err:
            comfy.samplers.KSampler = KSampler_orig
            raise err
        return ret

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
                break

        if target_line_number is None:
            raise FileNotFoundError
            # Target line not found, return the original function
            # return original_func

        # Insert the code to be injected after the target line
        lines.insert(target_line_number, item['code_to_insert'])

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