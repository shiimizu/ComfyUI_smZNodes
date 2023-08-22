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
import comfy.sdxl_clip
import comfy.sd1_clip
import comfy.sample
from comfy.sd1_clip import SD1Tokenizer, unescape_important, escape_important
from comfy.sdxl_clip import SDXLClipGTokenizer
from .modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedder2WithCustomWords
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
import itertools

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

# Open Clip
class FrozenOpenCLIPEmbedder2WithCustomWordsCustom(FrozenOpenCLIPEmbedder2WithCustomWords, SDXLClipGTokenizer):
    def populate_self_variables(self, from_):
        super_attrs = vars(from_)
        self_attrs = vars(self)
        self_attrs.update(super_attrs)

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
        parse_and_register_embeddings(self, line)
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


class FrozenCLIPEmbedderWithCustomWordsCustom(FrozenCLIPEmbedderForSDXLWithCustomWords, SD1Tokenizer):
    '''
    Custom class that also inherits a tokenizer to have the `_try_get_embedding()` method.
    '''
    def populate_self_variables(self, from_):
        super_attrs = vars(from_)
        self_attrs = vars(self)
        self_attrs.update(super_attrs)

    def __init__(self, wrapped: comfy.sd1_clip.SD1ClipModel, hijack):
        # okay to modiy since CLIP was cloned
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
        parse_and_register_embeddings(self, line) # register embeddings, discard return
        return super().tokenize_line(line)

    def tokenize(self, texts):
        tokenized = [self.tokenizer(text)["input_ids"][1:-1] for text in texts]
        return tokenized

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
                    self.hijack.embedding_db.register_embedding(Embedding(embed, embedding_name_verbose), self)
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

class PopulateVars():
    def populate_self_variables(self, from_):
        super_attrs = vars(from_)
        self_attrs = vars(self)
        self_attrs.update(super_attrs)

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

def sample_custom(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    params = locals()
    for idx, orig in enumerate([positive,negative]):
        if orig[0][1].get('from_smZ', False):
            if "comfy" not in orig[0][1]['params']['parser']:
                global opts
                cond_text = 'positive' if idx == 0 else 'negative'
                cid_orig = str(id(orig[0][1]['params']['clip']))
                orig[0][1]['params']['steps'] = steps
                conds_cache_cond = opts.conds_cache[cond_text].get(cid_orig, {})
                if conds_cache_cond != {}:
                    is_prompt_editing_ = False
                    schedules = getattr(conds_cache_cond[0][1]['pooled_output'], 'schedules', None)
                    if schedules!= None:
                        is_prompt_editing_ = is_prompt_editing(schedules)
                    conds_cache_cond[0][1]['params'].pop('step', None)
                    orig[0][1]['params'].pop('step', None)
                    if conds_cache_cond[0][1]['params'] == orig[0][1]['params'] or not is_prompt_editing_:
                        # print("Unchanged",cond_text, is_prompt_editing_)
                        params[cond_text] = conds_cache_cond
                        continue
                
                # re-encode with total number of steps
                # encode_fn = orig[0][1]['encode_fn']
                # cond = encode_fn(steps)[0]
                # cond[0][1]['encode_fn'] = orig[0][1]['encode_fn']
                cond = run(**orig[0][1]['params'])[0]
                cond[0][1]['params'] = {}
                cond[0][1]['params'].update(orig[0][1]['params'])
                cid = str(id(orig[0][1]['params']['clip']))
                opts.conds_cache[cond_text][cid] = cond

                params[cond_text] = cond

    return comfy.sample.sample_orig(**params)

from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
from nodes import CLIPTextEncode

def run(clip: comfy.sd.CLIP, text, parser, mean_normalization,
               multi_conditioning, use_old_emphasis_implementation,
               use_CFGDenoiser, with_SDXL, ascore, width, height, crop_w, 
               crop_h, target_width, target_height, text_g, text_l, steps=0, step=0, with_pooled=None):
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
    if parser == "comfy":
        clip_model_type_name = type(clip.cond_stage_model).__name__ 
        if with_SDXL and is_sdxl:
            if clip_model_type_name== "SDXLClipModel":
                out = CLIPTextEncodeSDXL().encode(clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)
                out[0][0][1]['aesthetic_score'] = sdxl_conds['aesthetic_score']
                return out
            elif clip_model_type_name == "SDXLRefinerClipModel":
                out = CLIPTextEncodeSDXLRefiner().encode(clip, ascore, width, height, text)
                for item in ['aesthetic_score', 'width', 'height', 'text_g', 'text_l']:
                    sdxl_conds.pop(item)
                out[0][0][1].update(sdxl_conds)
                return out
        else:
            return CLIPTextEncode().encode(clip, text)
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
            tokens = clip_clone.tokenize(text)
        cond = pooled = None

        # Because of prompt editing, we need the total number of steps
        # So this function will be called back at the sampling stage
        if steps != 0 or parser == "comfy++":
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

def encode_from_texts(clip: CLIP, texts, steps, return_pooled=False, multi=False, with_pooled=None):
    '''
    The function is our rendition of `clip.encode_from_tokens()`.
    It still calls `clip.encode_from_tokens()` but hijacks the
    `clip.cond_stage_model.encode_token_weights()` method
    so we can run our own version of `encode_token_weights()`

    Originally from `sd.py`: `encode_from_tokens()`
    '''
    tokens=texts
    partial_method = partial(get_learned_conditioning_custom, steps=steps, return_pooled=return_pooled, multi=multi, with_pooled=with_pooled)

    def assign_funcs(mc, fn=None):
        mc.encode_token_weights_orig = mc.encode_token_weights
        mc.encode_token_weights = MethodType(partial_method if fn == None else fn, mc)
    def restore_funcs(mc):
        if hasattr(mc, "encode_token_weights_orig"):
            mc.encode_token_weights = mc.encode_token_weights_orig

    # Get the max chunk_count to pad the tokens on varying clip_g and clip_l lengths.
    # To account for both positive and negative conditionings, opts.max_chunk_count is reset before sampling.
    def encode_token_weights_sdxl(self, token_weight_pairs):
        if opts.encode_count >= 2:
            opts.max_chunk_count = 0
            opts.encode_count = 0
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]

        # Only pad our initial gen of the schedules
        if with_pooled == None:
            if not opts.pad_with_repeats:
                opts.return_batch_chunks = True
                _, bcc_g = self.clip_g.encode_token_weights(token_weight_pairs_g)
                _, bcc_l = self.clip_l.encode_token_weights(token_weight_pairs_l)
                opts.return_batch_chunks = False
                opts.max_chunk_count = max(bcc_g, bcc_l, opts.max_chunk_count)
        else:
            if isinstance(with_pooled['schedules_'], dict): with_pooled['schedules_lg'] = with_pooled['schedules_']

        if with_pooled != None: with_pooled['schedules_'] = with_pooled['schedules_lg']['g']
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)

        if with_pooled != None: with_pooled['schedules_'] = with_pooled['schedules_lg']['l']
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)

        g_pooled.cond = {"g": g_pooled.cond, "l": l_pooled.cond }

        with devices.autocast(), torch.no_grad():
            if with_pooled == None:
                empty_g = self.clip_g.encode_token_weights([""])[0]
                empty_l = self.clip_l.encode_token_weights([""])[0]
                empty = torch.cat([empty_l, empty_g], dim=-1)
                shared.sd_model.cond_stage_model_empty_prompt_g = empty_g
                shared.sd_model.cond_stage_model_empty_prompt_l = empty_l
                shared.sd_model.cond_stage_model_empty_prompt = empty
        opts.encode_count += 1
        if opts.pad_with_repeats:
            # Both yield the same output
            g_out = expand(g_out, l_out, shared.sd_model.cond_stage_model_empty_prompt_g, True)
            l_out = expand(l_out, g_out, shared.sd_model.cond_stage_model_empty_prompt_l, True)
            # g_out = expand(g_out, l_out)
            # l_out = expand(l_out, g_out)
        return (torch.cat([l_out, g_out], dim=-1), g_pooled)

    class Context:
        def __init__(self):
            if "SDXL" in type(clip.cond_stage_model).__name__:
                if hasattr(clip.cond_stage_model, "clip_l"):
                    assign_funcs(clip.cond_stage_model.clip_l)
                if hasattr(clip.cond_stage_model, "clip_g"):
                    assign_funcs(clip.cond_stage_model.clip_g)
                if hasattr(clip.cond_stage_model, "clip_l") and hasattr(clip.cond_stage_model, "clip_g"):
                    assign_funcs(clip.cond_stage_model, encode_token_weights_sdxl)
            else:
                assign_funcs(clip.cond_stage_model)

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            if "SDXL" in type(clip.cond_stage_model).__name__:
                if hasattr(clip.cond_stage_model, "clip_l"):
                    restore_funcs(clip.cond_stage_model.clip_l)
                if hasattr(clip.cond_stage_model, "clip_g"):
                    restore_funcs(clip.cond_stage_model.clip_g)
                if hasattr(clip.cond_stage_model, "clip_l") and hasattr(clip.cond_stage_model, "clip_g"):
                    restore_funcs(clip.cond_stage_model)
            else:
                restore_funcs(clip.cond_stage_model)

    def encode_from_tokens(self, tokens, return_pooled=False):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

        # Only call this function on init
        if steps == 0:
            self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond
    with Context():
        return encode_from_tokens(clip, tokens, return_pooled)
        # return clip.encode_from_tokens(tokens, return_pooled)

def encode_from_tokens_with_custom_mean(clip: CLIP, tokens, return_pooled=False):
    '''
    The function is our rendition of `clip.encode_from_tokens()`.
    It still calls `clip.encode_from_tokens()` but hijacks the
    `clip.cond_stage_model.encode_token_weights()` method
    so we can run our own version of `encode_token_weights()`

    Originally from `sd.py`: `encode_from_tokens()`
    '''
    ret = None
    if "SDXL" in type(clip.cond_stage_model).__name__:
        encode_token_weights_orig_g = clip.cond_stage_model.clip_g.encode_token_weights
        encode_token_weights_orig_l = clip.cond_stage_model.clip_l.encode_token_weights
        try:
            clip.cond_stage_model.clip_g.encode_token_weights = MethodType(encode_token_weights_customized, clip.cond_stage_model.clip_g)
            clip.cond_stage_model.clip_l.encode_token_weights = MethodType(encode_token_weights_customized, clip.cond_stage_model.clip_l)
            ret = clip.encode_from_tokens(tokens, return_pooled)
            clip.cond_stage_model.clip_g.encode_token_weights = encode_token_weights_orig_g
            clip.cond_stage_model.clip_l.encode_token_weights = encode_token_weights_orig_l
        except Exception as error:
            clip.cond_stage_model.clip_g.encode_token_weights = encode_token_weights_orig_g
            clip.cond_stage_model.clip_l.encode_token_weights = encode_token_weights_orig_l
            raise error
    else:
        encode_token_weights_orig = clip.cond_stage_model.encode_token_weights
        try:
            clip.cond_stage_model.encode_token_weights = MethodType(encode_token_weights_customized, clip.cond_stage_model)
            ret = clip.encode_from_tokens(tokens, return_pooled)
            clip.cond_stage_model.encode_token_weights = encode_token_weights_orig
        except Exception as error:
            clip.cond_stage_model.encode_token_weights = encode_token_weights_orig
            raise error

    return ret

def encode_token_weights_customized(self: comfy.sd1_clip.SD1ClipModel|FrozenCLIPEmbedderWithCustomWordsBase, token_weight_pairs):
    if isinstance(self, comfy.sd1_clip.SD1ClipModel):
        to_encode = list(self.empty_tokens)
    else:
        to_encode = list(self.wrapped.empty_tokens)
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


# tokenize then encode
# from prompt_parser.py: get_learned_conditioning()
def get_learned_conditioning_custom(model: FrozenCLIPEmbedderWithCustomWordsBase, prompts: List[str], steps = 1, return_pooled=True, multi=False, with_pooled=None):
    first_pooled = None
    final_cond = None

    if with_pooled != None:
        first_pooled = with_pooled['pooled_output']
        schedules = with_pooled['schedules_']
        step = with_pooled['step_']
        conds_list = [[(0, 1.0)]]
        final_cond = reconstruct_schedules(schedules, step)
        if type(final_cond) == tuple:
            conds_list, final_cond = final_cond
        final_cond.cond = schedules
        first_pooled.cond = schedules
        with_pooled['conds_list_'] = conds_list
        return (final_cond, first_pooled) if return_pooled else final_cond

    # rewrite the functions here just to extract pooled

    if multi:
        res_indexes, prompt_flat_list, _prompt_indexes = prompt_parser.get_multicond_prompt_list(prompts)
        prompts = prompt_flat_list
        # learned_conditioning = prompt_parser.get_learned_conditioning(model, prompt_flat_list, steps)
    # The code below is from prompt_parser.get_learned_conditioning
    res = []
    prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}
    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        # debug(f'Prompt schedule: {prompt_schedule}')
        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue
        texts = prompt_parser.SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        # conds = model.get_learned_conditioning(texts)
        conds = model.forward(texts)
        if opts.return_batch_chunks and conds is not torch.Tensor:
            # It's safe to return early here since our input text is always one
            return conds
        if first_pooled == None:
            # first_pooled = conds.pooled
            if conds.pooled.shape[0] > 1:
                first_pooled = conds.pooled[1:2]
            else:
                first_pooled = conds.pooled[0:1]
        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]

            cond_schedule.append(prompt_parser.ScheduledPromptConditioning(end_at_step, cond))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)
    # first_pooled = first_pooled.cpu()
    if multi:
        res_mc = []
        learned_conditioning = res
        for indexes in res_indexes:
            res_mc.append([prompt_parser.ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])
        schedules = prompt_parser.MulticondLearnedConditioning(shape=(len(prompts),), batch=res_mc)
        conds_list, final_cond = prompt_parser.reconstruct_multicond_batch(schedules, 0)
        # final_cond = final_cond.cpu()
        final_cond.cond = schedules
        first_pooled.cond = schedules # for sdxl
    else:
        schedules=res
        final_cond = prompt_parser.reconstruct_cond_batch(schedules, 0)
        # final_cond = final_cond.cpu()
        final_cond.cond = schedules
        first_pooled.cond = schedules # for sdxl

    return (final_cond, first_pooled) if return_pooled else final_cond

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

def reconstruct_schedules(schedules, step):
    create_reconstruct_fn = lambda _cc: prompt_parser.reconstruct_multicond_batch if type(_cc).__name__ == "MulticondLearnedConditioning" else prompt_parser.reconstruct_cond_batch
    reconstruct_fn = create_reconstruct_fn(schedules)
    return reconstruct_fn(schedules, step)

class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ksampler = self._find_outer_instance()
        self.step = 0
        self.orig = comfy.samplers.CFGNoisePredictorOrig(model)
        self.inner_model = CFGDenoiser(model.apply_model)
        self.inner_model.num_timesteps = model.num_timesteps
        self.s_min_uncond = 0.0 # getattr(p, 's_min_uncond', 0.0)
        self.inner_model.device = self.ksampler.device if hasattr(self.ksampler, "device") else devices.device
        self.alphas_cumprod = model.alphas_cumprod
        self.cond_orig = None
        self.uncond_orig = None
        self.c_adm = None
        self.is_prompt_editing_u = False
        self.is_prompt_editing_c = False

    def _find_outer_instance(self):
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'self' in frame.f_locals and isinstance(frame.f_locals['self'], comfy.samplers.KSampler):
                return frame.f_locals['self']
            frame = frame.f_back
        return None

    def apply_model(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, model_options={}, seed=None):
        if not (cond[0][1].get('from_smZ', False) and uncond[0][1].get('from_smZ', False)):
            return self.orig.apply_model(x, timestep, cond, uncond, cond_scale, cond_concat, model_options, seed)

        if self.step == 0:
            self.init_cond = cond
            self.init_uncond = uncond
            self.is_prompt_editing_c = is_prompt_editing(getattr(cond[0][1]['pooled_output'], 'schedules', None))
            self.is_prompt_editing_u = is_prompt_editing(getattr(uncond[0][1]['pooled_output'], 'schedules', None))
        else:
            cond = self.init_cond
            cond_bak = cond
            if self.is_prompt_editing_c and cond[0][1].get('from_smZ', False) and "comfy" not in cond[0][1]['params']['parser']:
                # schedules = self.init_cond[0][1]['pooled_output'].schedules
                # multi_conditioning = self.init_cond[0][1]['params']['multi_conditioning']
                # self.clip.cond_stage_model.encode_token_weights = partial(self.clip.cond_stage_model.encode_token_weights, steps=steps, multi=multi_conditioning)
                # steps = self.init_cond[0][1]['params']['steps']
                params = self.init_cond[0][1]['params']
                params['step'] = self.step
                cond = run(with_pooled=self.init_cond[0][1], **params)[0] # params includes steps

                # encode_fn = self.init_cond[0][1]['encode_fn']
                # cond = encode_fn(self.init_cond[0][1]['params']['steps'], self.init_cond[0][1])[0]
                # cond = encode_fn(self.ksampler.steps, self.init_cond[0][1])[0]

                # schedules = getattr(cond[0][1]['pooled_output'], 'schedules', None)
                # cond = reconstruct_schedules(schedules, self.step)
                # if type(cond) == tuple:
                #     conds_list, cond = cond
                # cond = [[cond, cond_bak[0][1]]]
            # conds_list = [[(0, 1.0)]]
            
            uncond = self.init_uncond
            uncond_bak = uncond
            if self.is_prompt_editing_u and uncond[0][1].get('from_smZ', False) and "comfy" not in uncond[0][1]['params']['parser']:
                params = self.init_uncond[0][1]['params']
                params['step'] = self.step
                uncond = run(with_pooled=self.init_uncond[0][1], **params)[0]
                # encode_fn = self.init_uncond[0][1]['encode_fn']
                # uncond = encode_fn(self.init_uncond[0][1]['params']['steps'], self.init_uncond[0][1])[0]
                # uncond = encode_fn(self.ksampler.steps, self.init_uncond[0][1])[0]
                # schedules = getattr(uncond[0][1]['pooled_output'], 'schedules', None)
                # uncond = reconstruct_schedules(schedules, self.step)
                # if type(uncond) == tuple:
                #     _, uncond = uncond
                # uncond = [[uncond, uncond_bak[0][1]]]

            if (self.is_prompt_editing_c or self.is_prompt_editing_u) and (not torch.all(cond_bak[0][0] == cond[0][0]).item() or not torch.all(uncond_bak[0][0] == uncond[0][0]).item()):
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
            out = self.inner_model(x, timestep, cond=cond, uncond=unc, cond_scale=cond_scale, s_min_uncond=0.0, image_cond=image_cond)
        else:
            cond = [[co, cond[0][1]]]
            uncond = [[unc, uncond[0][1]]]
            out = self.orig.apply_model(x, timestep, cond, uncond, cond_scale, cond_concat, model_options, seed)
        self.step += 1
        return out

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