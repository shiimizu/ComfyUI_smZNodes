from __future__ import annotations
import os
import re
import torch
import inspect
import contextlib
import comfy
from functools import partial
import comfy.sdxl_clip
import comfy.sd1_clip
import comfy.sample
import comfy.utils
import comfy.samplers
from comfy.sd1_clip import unescape_important, escape_important, token_weights
from .modules import prompt_parser
from .modules.shared import SimpleNamespaceFast, logger, join_args
from .text_processing.textual_inversion import get_valid_embeddings, emb_re_
from .text_processing.classic_engine import ClassicTextProcessingEngine
from .text_processing.t5_engine import T5TextProcessingEngine

class Store(SimpleNamespaceFast): ...

store = Store()

def register_hooks():
    patches = [
        (comfy.samplers, 'calc_cond_batch', calc_cond_batch),
        (comfy.samplers, 'get_area_and_mult', get_area_and_mult),
        (comfy.samplers.KSamplerX0Inpaint, '__call__', KSamplerX0Inpaint___call__),
        (comfy.samplers.KSampler, 'sample', KSampler_sample),
        (comfy.samplers, 'sample', sample),
        (comfy.samplers.Sampler, 'max_denoise', max_denoise),
        (comfy.samplers, 'sampling_function', sampling_function),
    ]
    for parent, fn_name, fn_patch in patches:
        setattr(store, fn_patch.__name__, getattr(parent, fn_name))
        setattr(parent, fn_name, fn_patch)
    from .modules.rng import hook_prepare_noise
    hook_prepare_noise()

def iter_items(d):
    for key, value in d.items():
        yield key, value
        if isinstance(value, dict):
            yield from iter_items(value)

def find_nearest(a,b):
    # Calculate the absolute differences. 
    # the unsqueeze here broadcasts a to each row in b
    diff = torch.abs(a.unsqueeze(1) - b)

    # Find the indices of the nearest elements
    nearest_indices = torch.argmin(diff, dim=1)

    # Get the nearest elements from b
    nearest_elements = b[nearest_indices]
    return nearest_elements.to(a)

def get_value_from_args(fn, args, kwargs, key_to_lookup, idx=None):
    value = None
    if key_to_lookup in kwargs:
        value = kwargs[key_to_lookup]
    else:
        try:
            # Get its position in the formal parameters list and retrieve from args
            arg_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            index = arg_names.index(key_to_lookup)
            value = args[index] if index < len(args) else None
        except Exception:
            if idx is not None and idx < len(args):
                value = args[idx]
    return value

def calc_cond_batch(*args, **kwargs):
    x_in = kwargs['x_in'] if 'x_in' in kwargs else args[2]
    model_options = kwargs['model_options'] if 'model_options' in kwargs else args[4]
    if not hasattr(x_in, 'model_options'):
        x_in.model_options = model_options
    return store.calc_cond_batch(*args, **kwargs)

def get_area_and_mult(*args, **kwargs):
    x_in = kwargs['x_in'] if 'x_in' in kwargs else args[1]
    if (model_options:=getattr(x_in, 'model_options', None)) is not None and 'sigmas' in model_options:
        conds = kwargs['conds'] if 'conds' in kwargs else args[0]
        timestep_in = kwargs['timestep_in'] if 'timestep_in' in kwargs else args[2]
        sigmas = model_options['sigmas']
        ts_in = find_nearest(timestep_in, sigmas)
        cur_i = min(len(sigmas)-1, (sigmas==ts_in[0]).nonzero().item() + model_options.get('start_step', 0))
        if 'start_step' in conds and 'end_step' in conds:
            if not (cur_i >= conds['start_step'] and cur_i < conds['end_step']):
                return None
    return store.get_area_and_mult(*args, **kwargs)

def KSamplerX0Inpaint___call__(*args, **kwargs):
    self = kwargs['self'] if 'self' in kwargs else args[0]
    model_options = kwargs['model_options'] if 'model_options' in kwargs else args[4]
    model_options |= {'sigmas': self.sigmas}
    return store.KSamplerX0Inpaint___call__(*args, **kwargs)

def KSampler_sample(*args, **kwargs):
    orig_fn = store.KSampler_sample
    start_step = get_value_from_args(orig_fn, args, kwargs, 'start_step', 6)
    self = get_value_from_args(orig_fn, args, kwargs, 'self')
    if 'smZ_opts' in self.model_options:
        self.model_options['smZ_opts'].total_steps = self.steps
    if start_step is None:
        self.model_options.pop('start_step', None)
    else:
        self.model_options |= {'start_step': start_step}
    return orig_fn(*args, **kwargs)

def sample(*args, **kwargs):
    orig_fn = store.sample
    sampler = get_value_from_args(orig_fn, args, kwargs, 'sampler', 6)
    model_options = get_value_from_args(orig_fn, args, kwargs, 'model_options', 8)
    if 'smZ_opts' in model_options:
        if hasattr(sampler, 'sampler_function'):
            opts = model_options['smZ_opts']
            store.sampler_function = sampler.sampler_function
            sampler_function_sig_params = inspect.signature(sampler.sampler_function).parameters
            params = {x: getattr(opts, x)  for x in ['eta', 's_churn', 's_tmin', 's_tmax', 's_noise'] if x in sampler_function_sig_params}
            sampler.sampler_function = lambda *a, **kw: store.sampler_function(*a, **{**kw, **params})
    else:
        if hasattr(sampler, 'sampler_function') and hasattr(store, 'sampler_function'):
            sampler.sampler_function = store.sampler_function
    return orig_fn(*args, **kwargs)

def max_denoise(*args, **kwargs):
    orig_fn = store.max_denoise
    model_wrap = get_value_from_args(orig_fn, args, kwargs, 'model_wrap', 1)
    base_model = getattr(model_wrap, 'inner_model', None)
    model_options = getattr(model_wrap, 'model_options', getattr(base_model, 'model_options', None))
    res = orig_fn(*args, **kwargs)
    res = getattr(model_options.get('smZ_opts', res), 'sgm_noise_multiplier', res)
    return res

def sampling_function(*args, **kwargs):
    orig_fn = store.sampling_function
    model_options = get_value_from_args(orig_fn, args, kwargs, 'model_options', 6)
    if 'smZ_opts' in model_options and 'sigmas' in model_options:
        opts = model_options['smZ_opts']
        if opts.s_min_uncond_all or opts.s_min_uncond > 0 or opts.skip_early_cond > 0:
            cond_scale = _cond_scale = get_value_from_args(orig_fn, args, kwargs, 'cond_scale', 5)
            sigmas = model_options['sigmas']
            sigma = get_value_from_args(orig_fn, args, kwargs, 'timestep', 2)
            ts_in = find_nearest(sigma, sigmas)
            step = min(len(sigmas)-1, (sigmas==ts_in[0]).nonzero().item() + model_options.get('start_step', 0))
            total_steps = getattr(opts, 'total_steps', len(sigmas))
            skip_uncond = False

            if opts.skip_early_cond > 0 and step / total_steps <= opts.skip_early_cond:
                skip_uncond = True
            elif (step % 2 or opts.s_min_uncond_all) and opts.s_min_uncond > 0 and sigma[0] < opts.s_min_uncond:
                skip_uncond = True

            if skip_uncond:
                cond_scale = 1.0
            
            if _cond_scale != cond_scale:
                if 'cond_scale' not in kwargs:
                    args = args[:5]
                kwargs['cond_scale'] = cond_scale
    return orig_fn(*args, **kwargs)

@contextlib.contextmanager
def HijackClip(clip, mean_normalization):
    a1 = 'tokenizer', 'tokenize_with_weights'
    a2 = 'cond_stage_model', 'encode_token_weights'
    ls = [a1, a2]
    store = {}
    store_orig = {}
    try:
        for obj, attr in ls:
            for clip_name, v in iter_items(getattr(clip, obj).__dict__):
                if hasattr(v, attr):
                    logger.debug(join_args(attr, obj, clip_name, type(v), getattr(v, attr).__qualname__))
                    if clip_name not in store_orig:
                        store_orig[clip_name] = {}
                    store_orig[clip_name][obj] = v
        for clip_name, inner_store in store_orig.items():
            text_encoder = inner_store['cond_stage_model']
            tokenizer = inner_store['tokenizer']
            emphasis_name = 'Original' if mean_normalization else "No norm"
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
            store[clip_name] = text_processing_engine
            for obj, attr in ls:
                setattr(inner_store[obj], attr, getattr(store[clip_name], attr))
        yield clip
    finally:
        for clip_name, inner_store in store_orig.items():
            getattr(inner_store[a2[0]], a2[1]).__self__.unhook()
            for obj, attr in ls:
                delattr(inner_store[obj], attr)
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
                    logger.debug(join_args(attr, obj, clip_name, type(v), getattr(v, attr).__qualname__))
                    if clip_name not in store_orig:
                        store_orig[clip_name] = {}
                    store_orig[clip_name][obj] = v
                    for obj, attr in ls:
                        setattr(v, attr, partial(tokenize_with_weights_custom, v))
        yield clip
    finally:
        for clip_name, inner_store in store_orig.items():
            for obj, attr in ls:
                delattr(inner_store[obj], attr)
        del store_orig


def transform_schedules(schedules, with_steps=False):
    end_steps = [schedule.end_at_step for schedule in schedules]
    start_end_pairs = list(zip([0] + end_steps[:-1], end_steps))
    
    return [
        [
            schedule.cond.pop("cond", None),
            schedule.cond | {
                "start_step": start_step,
                "end_step": end_step
            } if with_steps else schedule.cond
        ]
        for schedule, (start_step, end_step) in zip(schedules, start_end_pairs)
    ]

def convert_schedules_to_comfy(schedules, with_steps=False):
    return [transform_schedules(sublist, with_steps) for sublist in schedules]

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
                        logger.debug(f'using embedding:{embedding_name}')
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