from pathlib import Path
import os
import shutil
import subprocess
import importlib
import inspect
from functools import partial

def install(module):
    import sys
    try:
        print(f"\033[92m[smZNodes] \033[0;31m{module} is not installed. Attempting to install...\033[0m")
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
        reload()
        print(f"\033[92m[smZNodes] {module} Installed!\033[0m")
    except:
        print(f"\033[92m[smZNodes] \033[0;31mFailed to install {module}.\033[0m")

# Reload modules after installation
PRELOADED_MODULES = set()
def init() :
    # local imports to keep things neat
    from sys import modules
    import importlib
    global PRELOADED_MODULES
    # sys and importlib are ignored here too
    PRELOADED_MODULES = set(modules.values())
def reload() :
    from sys import modules
    import importlib
    for module in set(modules.values()) - PRELOADED_MODULES :
        try :
            importlib.reload(module)
        except :
            pass
init()

# compel =================
if importlib.util.find_spec('compel') is None:
    install("compel")

# lark =================
if importlib.util.find_spec('lark') is None:
    install("lark")
# ============================

WEB_DIRECTORY = "./web"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ==== web ======
cwd_path = Path(__file__).parent
comfy_path = cwd_path.parent.parent

def setup_web_extension():
    import nodes
    web_extension_path = os.path.join(comfy_path, "web", "extensions", "smZNodes")

    if os.path.exists(web_extension_path):
        shutil.rmtree(web_extension_path)
    if not hasattr(nodes, "EXTENSION_WEB_DIRS"):
        # print(f"[smZNodes]: Your ComfyUI version is outdated. Please update to the latest version.")
        # setup js
        if not os.path.exists(web_extension_path):
            os.makedirs(web_extension_path)

        js_src_path = os.path.join(cwd_path, "web/js", "smZdynamicWidgets.js")
        shutil.copy(js_src_path, web_extension_path)

setup_web_extension()

# ==============

# add_sample_dpmpp_2m_alt, inject_code, opts as smZ_opts
from .smZNodes import add_sample_dpmpp_2m_alt, inject_code, CFGNoisePredictor

add_sample_dpmpp_2m_alt()

# ==============
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

import comfy
if not hasattr(comfy.samplers, 'Sampler'):
    print(f"[smZNodes]: Your ComfyUI version is outdated. Please update to the latest version.")
    comfy.samplers.KSampler.sample = inject_code(comfy.samplers.KSampler.sample, payload)
else:
    _KSampler_sample = comfy.samplers.KSampler.sample
    _Sampler = comfy.samplers.Sampler
    _max_denoise = comfy.samplers.Sampler.max_denoise
    _sample = comfy.samplers.sample
    _wrap_model = comfy.samplers.wrap_model

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
                sampler.sampler_function = partial(sampler.sampler_function_orig, **params)
        model.model_options = model_options # Add model_options to CFGNoisePredictor
        return _sample(*args, **kwargs)

    class Sampler(_Sampler):
        def max_denoise(self, model_wrap: CFGNoisePredictor, sigmas):
            base_model = model_wrap.inner_model
            res = _max_denoise(self, model_wrap, sigmas)
            if (model_options:=base_model.model_options) is not None:
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
comfy.samplers.CFGNoisePredictor = CFGNoisePredictor

if not hasattr(comfy.sample, 'prepare_noise_orig'):
    comfy.sample.prepare_noise_orig = comfy.sample.prepare_noise

if hasattr(comfy.model_management, 'unet_dtype'):
    if not hasattr(comfy.model_management, 'unet_dtype_orig'):
        comfy.model_management.unet_dtype_orig = comfy.model_management.unet_dtype
    from .modules import devices
    def unet_dtype(device=None, model_params=0):
        dtype = comfy.model_management.unet_dtype_orig(device=device, model_params=model_params)
        if model_params != 0:
            devices.dtype_unet = dtype
        return dtype
    comfy.model_management.unet_dtype = unet_dtype
