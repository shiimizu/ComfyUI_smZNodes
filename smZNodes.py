from __future__ import annotations
import re
import torch
import inspect
import contextlib
import logging
import comfy
import math
import ctypes
from decimal import Decimal
from functools import partial
from random import getrandbits
import comfy.sdxl_clip
import comfy.sd1_clip
import comfy.sample
import comfy.utils
import comfy.samplers
from comfy.sd1_clip import unescape_important, escape_important, token_weights
from .modules.shared import SimpleNamespaceFast, Options, logger, join_args
from .modules.text_processing import prompt_parser
from .modules.text_processing.past_classic_engine import process_texts_past
from .modules.text_processing.textual_inversion import EmbbeddingRegex
from .modules.text_processing.classic_engine import ClassicTextProcessingEngine
from .modules.text_processing.t5_engine import T5TextProcessingEngine

class Store(SimpleNamespaceFast): ...

store = Store()

def register_hooks():
    from .modules.rng import prepare_noise
    patches = [
        (comfy.samplers, 'get_area_and_mult', get_area_and_mult),
        (comfy.samplers.KSampler, 'sample', KSampler_sample),
        (comfy.samplers.KSAMPLER, 'sample', KSAMPLER_sample),
        (comfy.samplers, 'sample', sample),
        (comfy.samplers.Sampler, 'max_denoise', max_denoise),
        (comfy.samplers, 'sampling_function', sampling_function),
        (comfy.sample, 'prepare_noise', prepare_noise),
    ]
    for parent, fn_name, fn_patch in patches:
        if not hasattr(store, fn_patch.__name__):
            setattr(store, fn_patch.__name__, getattr(parent, fn_name))
            setattr(parent, fn_name, fn_patch)

def iter_items(d):
    for key, value in d.items():
        yield key, value
        if isinstance(value, dict):
            yield from iter_items(value)

def find_nearest(a,b):
    # Calculate the absolute differences. 
    diff = (a - b).abs()

    # Find the indices of the nearest elements
    nearest_indices = diff.argmin()

    # Get the nearest elements from b
    return b[nearest_indices]

def get_area_and_mult(*args, **kwargs):
    conds = args[0]
    if 'start_perc' in conds and 'end_perc' in conds and "init_steps" in conds:
        timestep_in = args[2]
        sigmas = store.sigmas
        if conds['init_steps'] == sigmas.shape[0] - 1:
            total = Decimal(sigmas.shape[0] - 1)
        else:
            sigmas_ = store.sigmas.unique(sorted=True).sort(descending=True)[0]
            if len(sigmas) == len(sigmas_):
                # Sampler Custom with sigmas: no change
                total = Decimal(sigmas.shape[0] - 1)
            else:
                # Sampler with restarts: dedup the sigmas and add one
                sigmas = sigmas_
                total = Decimal(sigmas.shape[0] + 1)
        ts_in = find_nearest(timestep_in, sigmas)
        cur_i = ss[0].item() if (ss:=(sigmas == ts_in).nonzero()).shape[0] != 0 else 0
        cur = Decimal(cur_i) / total
        start = conds['start_perc']
        end = conds['end_perc']
        if not (cur >= start and cur < end):
            return None
    return store.get_area_and_mult(*args, **kwargs)

def KSAMPLER_sample(*args, **kwargs):
    orig_fn = store.KSAMPLER_sample
    extra_args = None
    model_options = None
    try:
        extra_args = kwargs['extra_args'] if 'extra_args' in kwargs else args[3]
        model_options = extra_args['model_options']
    except Exception: ...
    if model_options is not None and extra_args is not None:
        sigmas_ = kwargs['sigmas'] if 'sigmas' in kwargs else args[2]
        sigmas_all = model_options.pop('sigmas', None)
        sigmas = sigmas_all if sigmas_all is not None else sigmas_
        store.sigmas = sigmas

    import comfy.k_diffusion.sampling
    if hasattr(comfy.k_diffusion.sampling, 'default_noise_sampler_orig'):
        if getattr(comfy.k_diffusion.sampling.default_noise_sampler, 'init', False):
            comfy.k_diffusion.sampling.default_noise_sampler.init = False
        else:
            comfy.k_diffusion.sampling.default_noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler_orig

    if 'Hijack' in comfy.k_diffusion.sampling.torch.__class__.__name__:
        if getattr(comfy.k_diffusion.sampling.torch, 'init', False):
            comfy.k_diffusion.sampling.torch.init = False
        else:
            if hasattr(comfy.k_diffusion.sampling, 'torch_orig'):
                comfy.k_diffusion.sampling.torch = comfy.k_diffusion.sampling.torch_orig
    return orig_fn(*args, **kwargs)

def KSampler_sample(*args, **kwargs):
    orig_fn = store.KSampler_sample
    self = args[0]
    model_patcher = getattr(self, 'model', None)
    model_options = getattr(model_patcher, 'model_options', None)
    if model_options is not None:
        sigmas = None
        try: sigmas = kwargs['sigmas'] if 'sigmas' in kwargs else args[10]
        except Exception: ...
        if sigmas is None:
            sigmas = getattr(self, 'sigmas', None)
        if sigmas is not None:
            model_options = model_options.copy()
            model_options['sigmas'] = sigmas
            self.model.model_options = model_options
    return orig_fn(*args, **kwargs)

def sample(*args, **kwargs):
    orig_fn = store.sample
    model_patcher = args[0]
    model_options = getattr(model_patcher, 'model_options', None)
    sampler = kwargs['sampler'] if 'sampler' in kwargs else args[6]
    if model_options is not None and Options.KEY in model_options:
        if hasattr(sampler, 'sampler_function'):
            opts = model_options[Options.KEY]
            if not hasattr(sampler, f'_sampler_function'):
                sampler._sampler_function = sampler.sampler_function
            sampler_function_sig_params = inspect.signature(sampler._sampler_function).parameters
            params = {x: getattr(opts, x)  for x in ['eta', 's_churn', 's_tmin', 's_tmax', 's_noise'] if x in sampler_function_sig_params}
            sampler.sampler_function = lambda *a, **kw: sampler._sampler_function(*a, **{**kw, **params})
    else:
        if hasattr(sampler, '_sampler_function'):
            sampler.sampler_function = sampler._sampler_function
    return orig_fn(*args, **kwargs)

def max_denoise(*args, **kwargs):
    orig_fn = store.max_denoise
    model_wrap = kwargs['model_wrap'] if 'model_wrap' in kwargs else args[1]
    base_model = getattr(model_wrap, 'inner_model', None)
    model_options = getattr(model_wrap, 'model_options', getattr(base_model, 'model_options', None))
    return orig_fn(*args, **kwargs) if getattr(model_options.get(Options.KEY, True), 'sgm_noise_multiplier', True) else False

def sampling_function(*args, **kwargs):
    orig_fn = store.sampling_function
    model_options = kwargs['model_options'] if 'model_options' in kwargs else args[6]
    model_options=model_options.copy()
    kwargs['model_options'] = model_options
    if Options.KEY in model_options:
        opts = model_options[Options.KEY]
        if opts.s_min_uncond_all or opts.s_min_uncond > 0 or opts.skip_early_cond > 0:
            cond_scale = _cond_scale = kwargs['cond_scale'] if 'cond_scale' in kwargs else args[5]
            sigmas = store.sigmas
            sigma = kwargs['timestep'] if 'timestep' in kwargs else args[2]
            ts_in = find_nearest(sigma, sigmas)
            step = ss[0].item() if (ss:=(sigmas == ts_in).nonzero()).shape[0] != 0 else 0
            total_steps = sigmas.shape[0] - 1

            if opts.skip_early_cond > 0 and step / total_steps <= opts.skip_early_cond:
                cond_scale = 1.0
            elif (step % 2 or opts.s_min_uncond_all) and opts.s_min_uncond > 0 and sigma[0] < opts.s_min_uncond:
                cond_scale = 1.0

            if cond_scale != _cond_scale:
                if 'cond_scale' not in kwargs:
                    args = args[:5]
                kwargs['cond_scale'] = cond_scale

    cond = kwargs['cond'] if 'cond' in kwargs else args[4]
    weights = [x.get('weight', None) for x in cond]
    has_some = any(item is not None for item in weights) and len(weights) > 1
    if has_some:
        out = CFGDenoiser(orig_fn).sampling_function(*args, **kwargs)
    else:
        out = orig_fn(*args, **kwargs)
    return out


@contextlib.contextmanager
def HijackClip(clip, opts):
    a1 = 'tokenizer', 'tokenize_with_weights'
    a2 = 'cond_stage_model', 'encode_token_weights'
    ls = [a1, a2]
    store = {}
    store_orig = {}
    try:
        for obj, attr in ls:
            for clip_name, v in iter_items(getattr(clip, obj).__dict__):
                if hasattr(v, attr):
                    logger.debug(join_args(attr, obj, clip_name, type(v).__qualname__, getattr(v, attr).__qualname__))
                    if clip_name not in store_orig:
                        store_orig[clip_name] = {}
                    store_orig[clip_name][obj] = v
        for clip_name, inner_store in store_orig.items():
            text_encoder = inner_store['cond_stage_model']
            tokenizer = inner_store['tokenizer']
            emphasis_name = 'Original' if opts.prompt_mean_norm else "No norm"
            if 't5' in clip_name:
                text_processing_engine = T5TextProcessingEngine(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    emphasis_name=emphasis_name,
                )
            else:
                text_processing_engine = ClassicTextProcessingEngine(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    emphasis_name=emphasis_name,
                )
            text_processing_engine.opts = opts
            text_processing_engine.process_texts_past = partial(process_texts_past, text_processing_engine)
            store[clip_name] = text_processing_engine
            for obj, attr in ls:
                setattr(inner_store[obj], attr, getattr(store[clip_name], attr))
        yield clip
    finally:
        for clip_name, inner_store in store_orig.items():
            getattr(inner_store[a2[0]], a2[1]).__self__.unhook()
            for obj, attr in ls:
                try: delattr(inner_store[obj], attr)
                except Exception: ...
        del store
        del store_orig

@contextlib.contextmanager
def HijackClipComfy(clip):
    a1 = 'tokenizer', 'tokenize_with_weights'
    ls = [a1]
    store_orig = {}
    try:
        for obj, attr in ls:
            for clip_name, v in iter_items(getattr(clip, obj).__dict__):
                if hasattr(v, attr):
                    logger.debug(join_args(attr, obj, clip_name, type(v).__qualname__, getattr(v, attr).__qualname__))
                    if clip_name not in store_orig:
                        store_orig[clip_name] = {}
                    store_orig[clip_name][obj] = v
                    setattr(v, attr, partial(tokenize_with_weights_custom, v))
        yield clip
    finally:
        for clip_name, inner_store in store_orig.items():
            for obj, attr in ls:
                try: delattr(inner_store[obj], attr)
                except Exception: ...
        del store_orig

def transform_schedules(steps, schedules, weight=None, with_weight=False):
    end_steps = [schedule.end_at_step for schedule in schedules]
    start_end_pairs = list(zip([0] + end_steps[:-1], end_steps))
    with_prompt_editing = len(schedules) > 1

    def process(schedule, start_step, end_step):
        nonlocal with_prompt_editing
        d = schedule.cond.copy()
        d.pop('cond', None)
        if with_prompt_editing:
            d |= {"start_perc": Decimal(start_step) / Decimal(steps), "end_perc": Decimal(end_step)  / Decimal(steps), "init_steps": steps}
        if weight is not None and with_weight:
            d['weight'] = weight
        return d
    return [
        [
            schedule.cond.get("cond", None),
            process(schedule, start_step, end_step)
        ]
        for schedule, (start_step, end_step) in zip(schedules, start_end_pairs)
    ]

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def convert_schedules_to_comfy(schedules, steps, multi=False):
    if multi:
        out = [[transform_schedules(steps, x.schedules, x.weight, len(batch)>1) for x in batch] for batch in schedules.batch]
        out = flatten(out)
    else:
        out = [transform_schedules(steps, sublist) for sublist in schedules]
    return flatten(out)

def get_learned_conditioning(model, prompts, steps, multi=False, *args, **kwargs):
    if multi:
        schedules = prompt_parser.get_multicond_learned_conditioning(model, prompts, steps, *args, **kwargs)
    else:
        schedules = prompt_parser.get_learned_conditioning(model, prompts, steps, *args, **kwargs)
    schedules_c = convert_schedules_to_comfy(schedules, steps, multi)
    return schedules_c

class CustomList(list):
    def __init__(self, *args):
        super().__init__(*args)
    def __setattr__(self, name: str, value: re.Any):
        super().__setattr__(name, value)
        return self

def modify_locals_values(frame, fn):
    # https://stackoverflow.com/a/34671307
    try: ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
    except Exception: ...
    fn(frame)
    try: ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
    except Exception: ...

def update_locals(frame,k,v,list_app=False):
    if not list_app:
        modify_locals_values(frame, lambda _frame: _frame.f_locals.__setitem__(k, v))
    else:
        if not isinstance(frame.f_locals[k], CustomList):
            out_conds_store = CustomList([])
            out_conds_store.outputs=[]
            modify_locals_values(frame, lambda _frame: _frame.f_locals.__setitem__(k, out_conds_store))
        v.area = frame.f_locals['area']
        v.mult = frame.f_locals['mult']
        frame.f_locals[k].outputs.append(v)
        frame.f_locals[k].out_conds = frame.f_locals['out_conds']
        frame.f_locals[k].out_counts = frame.f_locals['out_counts']
        modify_locals_values(frame, lambda _frame: _frame.f_locals.__setitem__('batch_chunks', 0))

def model_function_wrapper_cd(model, args, id, model_options={}):
    input_x = args['input']
    timestep_ = args['timestep']
    c = args['c']
    cond_or_uncond = args['cond_or_uncond']
    batch_chunks = len(cond_or_uncond)
    if f'model_function_wrapper_{id}' in model_options:
        output = model_options[f'model_function_wrapper_{id}'](model, args)
    else:
        output = model(input_x, timestep_, **c)
    output.cond_or_uncond = cond_or_uncond
    output.batch_chunks = batch_chunks
    output.output_chunks = output.chunk(batch_chunks)
    output.chunk = lambda *aa, **kw: output
    get_parent_variable('out_conds', list, lambda frame: update_locals(frame, 'out_conds', output, list_app=True))
    return output

def get_parent_variable(vname, vtype, fn):
    frame = inspect.currentframe().f_back  # Get the current frame's parent
    while frame:
        if vname in frame.f_locals:
            val = frame.f_locals[vname]
            if isinstance(val, vtype):
                if fn is not None:
                    fn(frame)
                return frame.f_locals[vname]
        frame = frame.f_back
    return None

def cd_cfg_function(kwargs, id):
    model_options = kwargs['model_options']
    if f"sampler_cfg_function_{id}" in model_options:
        return model_options[f'sampler_cfg_function_{id}'](kwargs)
    x = kwargs['input']
    cond_pred = kwargs['cond_denoised']
    uncond_pred = kwargs['uncond_denoised']
    cond_scale = kwargs['cond_scale']
    cfg_result = model_options['cfg_result']
    cfg_result += (cond_pred - uncond_pred) * cond_scale
    return x - cfg_result

class CFGDenoiser:
    def __init__(self, orig_fn) -> None:
        self.orig_fn = orig_fn

    def sampling_function(self, model, x, timestep, uncond, cond, cond_scale, model_options, *args0, **kwargs0):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        else:
            uncond_ = uncond

        conds = [cond, uncond_]

        if uncond_ is None:
            return self.orig_fn(model, x, timestep, uncond, cond, cond_scale, model_options, *args0, **kwargs0)

        id = getrandbits(7)
        if 'model_function_wrapper' in model_options:
            model_options[f'model_function_wrapper_{id}'] = model_options.pop('model_function_wrapper')
        model_options['model_function_wrapper'] = partial(model_function_wrapper_cd, id=id, model_options=model_options)
        out = comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)
        model_options.pop('model_function_wrapper', None)
        if f'model_function_wrapper_{id}' in model_options:
            model_options['model_function_wrapper'] = model_options.pop(f'model_function_wrapper_{id}')
        
        outputs = out.outputs

        out_conds = out.out_conds
        out_counts= out.out_counts
        if len(out_conds) < len(out_counts):
            for _ in out_counts:
                out_conds.append(torch.zeros_like(outputs[0].output_chunks[0]))

        oconds=[]
        for _output in outputs:
            cond_or_uncond=_output.cond_or_uncond
            batch_chunks=_output.batch_chunks
            output=_output.output_chunks
            area=_output.area
            mult=_output.mult
            for o in range(batch_chunks):
                cond_index = cond_or_uncond[o]
                a = area[o]
                if a is None:
                    if cond_index == 0:
                        oconds.append(output[o] * mult[o])
                    else:
                        out_conds[cond_index] += output[o] * mult[o]
                        out_counts[cond_index] += mult[o]
                else:
                    out_c = out_conds[cond_index] if cond_index != 0 else torch.zeros_like(out_conds[cond_index])
                    out_cts = out_counts[cond_index]
                    dims = len(a) // 2
                    for i in range(dims):
                        out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                        out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                    out_c += output[o] * mult[o]
                    out_cts += mult[o]
                    if cond_index == 0:
                        oconds.append(out_c)

        for i in range(len(out_conds)):
            if i != 0: 
                out_conds[i] /= out_counts[i]

        del out
        out = out_conds

        for fn in model_options.get("sampler_pre_cfg_function", []):
            out[0] = torch.cat(oconds).to(oconds[0])
            args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                    "input": x, "sigma": timestep, "model": model, "model_options": model_options}
            out  = fn(args)

        # ComfyUI: last prompt -> first
        # conds were reversed in calc_cond_batch, so do the same for weights
        weights = [x.get('weight', None) for x in cond]
        weights.reverse()
        out_uncond = out[1]
        cfg_result = out_uncond.clone()
        cond_scale = cond_scale / max(len(oconds), 1)
        
        if "sampler_cfg_function" in model_options:
            model_options[f'sampler_cfg_function_{id}'] = model_options.pop('sampler_cfg_function')
        model_options['sampler_cfg_function'] = partial(cd_cfg_function, id=id)
        model_options['cfg_result'] = cfg_result

        # ComfyUI: computes the average -> do cfg
        # A1111: (cond - uncond) / total_len_of_conds -> in-place addition for each cond -> results in cfg
        for ix, ocond in enumerate(oconds):
            weight = (weights[ix:ix+1] or [1.0])[0] or 1.0
            # cfg_result += (ocond - out_uncond) * (weight * cond_scale) # all this code just to do this
            if f"sampler_cfg_function_{id}" in model_options:
                # case when there's another cfg_fn. subtract out_uncond and in-place add the result. feed result back in.
                cfg_result += comfy.samplers.cfg_function(model, ocond, out_uncond, weight * cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_) - out_uncond
            else: # calls cd_cfg_function and does an in-place addition
                if model_options.get("sampler_post_cfg_function", []):
                    # feed the result back in.
                    cfg_result = comfy.samplers.cfg_function(model, ocond, out_uncond, weight * cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)
                else:
                    # default case. discards the output.
                    comfy.samplers.cfg_function(model, ocond, out_uncond, weight * cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)
            model_options['cfg_result'] = cfg_result
        return cfg_result

def tokenize_with_weights_custom(self, text:str, return_word_ids=False, tokenizer_options={}, **kwargs):
    '''
    Takes a prompt and converts it to a list of (token, weight, word id) elements.
    Tokens can both be integer tokens and pre computed CLIP tensors.
    Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
    Returned list has the dimensions NxM where M is the input size of CLIP
    '''
    min_length = tokenizer_options.get("{}_min_length".format(self.embedding_key), self.min_length)
    min_padding = tokenizer_options.get("{}_min_padding".format(self.embedding_key), self.min_padding)

    text = escape_important(text)
    parsed_weights = token_weights(text, 1.0)
    embr = EmbbeddingRegex(self.embedding_directory)

    # tokenize words
    tokens = []
    for weighted_segment, weight in parsed_weights:
        to_tokenize = unescape_important(weighted_segment)
        split = re.split(' {0}|\n{0}'.format(self.embedding_identifier), to_tokenize)
        to_tokenize = [split[0]]
        for i in range(1, len(split)):
            to_tokenize.append("{}{}".format(self.embedding_identifier, split[i]))

        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            matches = embr.pattern.finditer(word)
            last_end = 0
            leftovers=[]
            for _, match in enumerate(matches, start=1):
                start=match.start()
                end_match=match.end()
                if (fragment:=word[last_end:start]):
                    leftovers.append(fragment)
                ext = (match.group(4) or (match.group(3) or ''))
                embedding_sname = (match.group(2) or '').removesuffix(ext)
                embedding_name = embedding_sname + ext
                if embedding_name:
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        logger.debug(f'using embedding:{embedding_name}')
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                last_end = end_match
            if (fragment:=word[last_end:]):
                leftovers.append(fragment)
                word_new = ''.join(leftovers)
                end = 999999999999
                if self.tokenizer_adds_end_token:
                    end = -1
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word_new)["input_ids"][self.tokens_start:end]])

    #reshape token array to CLIP input size
    batched_tokens = []
    batch = []
    if self.start_token is not None:
        batch.append((self.start_token, 1.0, 0))
    batched_tokens.append(batch)
    for i, t_group in enumerate(tokens):
        #determine if we're going to try and keep the tokens in a single batch
        is_large = len(t_group) >= self.max_word_length
        if self.end_token is not None:
            has_end_token = 1
        else:
            has_end_token = 0

        while len(t_group) > 0:
            if len(t_group) + len(batch) > self.max_length - has_end_token:
                remaining_length = self.max_length - len(batch) - has_end_token
                #break word in two and add end token
                if is_large:
                    batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                    if self.end_token is not None:
                        batch.append((self.end_token, 1.0, 0))
                    t_group = t_group[remaining_length:]
                #add end token and pad
                else:
                    if self.end_token is not None:
                        batch.append((self.end_token, 1.0, 0))
                    if self.pad_to_max_length:
                        batch.extend([(self.pad_token, 1.0, 0)] * (remaining_length))
                #start new batch
                batch = []
                if self.start_token is not None:
                    batch.append((self.start_token, 1.0, 0))
                batched_tokens.append(batch)
            else:
                batch.extend([(t,w,i+1) for t,w in t_group])
                t_group = []

    #fill last batch
    if self.end_token is not None:
        batch.append((self.end_token, 1.0, 0))
    if min_padding is not None:
        batch.extend([(self.pad_token, 1.0, 0)] * min_padding)
    if self.pad_to_max_length and len(batch) < self.max_length:
        batch.extend([(self.pad_token, 1.0, 0)] * (self.max_length - len(batch)))
    if min_length is not None and len(batch) < min_length:
        batch.extend([(self.pad_token, 1.0, 0)] * (min_length - len(batch)))

    if not return_word_ids:
        batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

    return batched_tokens

# ========================================================================

from server import PromptServer

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

def prompt_handler(json_data):
    data=json_data['prompt']
    steps_validator = lambda x: isinstance(x, (int, float, str))
    text_validator = lambda x: isinstance(x, str)
    def find_nearest_ksampler(clip_id):
        """Find the nearest KSampler node that references the given CLIPTextEncode id."""
        nonlocal data, steps_validator
        for ksampler_id, node in data.items():
            if "class_type" in node and ("Sampler" in node["class_type"] or "sampler" in node["class_type"]):
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
                    s = 1
                    try: s = min(max(1, int(steps_input_value)), 10000)
                    except Exception as e:
                        logging.warning(f"\033[33mWarning:\033[0m [smZNodes] Skipping prompt editing. Try recreating the node. {e}")
                    return s
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
        if "class_type" in node and node["class_type"] == "smZ CLIPTextEncode":
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
                # logger.debug(f'id: {clip_id} | steps: {steps}')
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
        except Exception: ...

def add_custom_samplers():
    samplers = [
        add_sample_dpmpp_2m_alt,
    ]
    for add_sampler in samplers:
        add_sampler()
