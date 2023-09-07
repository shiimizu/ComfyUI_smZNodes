from pathlib import Path
import os
import shutil
import subprocess
import importlib

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

# ==== web ======
cwd_path = Path(__file__).parent
comfy_path = cwd_path.parent.parent
def copy_to_web(file):
    """Copy a file to the web extension path."""
    shutil.copy(file, web_extension_path)

web_extension_path = os.path.join(comfy_path, "web", "extensions", "smZNodes")

if os.path.exists(web_extension_path):
    shutil.rmtree(web_extension_path)

# ==============
WEB_DIRECTORY = "./web"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# add_sample_dpmpp_2m_alt, inject_code, opts as smZ_opts
from .smZNodes import add_sample_dpmpp_2m_alt, inject_code, CFGNoisePredictor

add_sample_dpmpp_2m_alt()
import comfy.sample
from comfy.samplers import KSampler
payload = [{
    "target_line": 'extra_args["denoise_mask"] = denoise_mask',
    "code_to_insert": """
            if (any([_p[1].get('from_smZ', False) for _p in positive]) or 
                any([_p[1].get('from_smZ', False) for _p in negative])):
                from ComfyUI_smZNodes.modules.shared import opts as smZ_opts
                if not smZ_opts.sgm_noise_multiplier:
                    max_denoise = False
"""
},
{
    "target_line": 'positive = positive[:]',
    "code_to_insert": """
        if (any([_p[1].get('from_smZ', False) for _p in positive]) or 
            any([_p[1].get('from_smZ', False) for _p in negative])):
            from ComfyUI_smZNodes.modules.shared import opts as smZ_opts
            smZ_opts.noise = noise
        self.model_denoise.step = start_step if start_step != None else 0
"""
},
]

KSampler.sample = inject_code(KSampler.sample, payload)

import comfy
comfy.samplers.CFGNoisePredictorOrig = comfy.samplers.CFGNoisePredictor
comfy.samplers.CFGNoisePredictor = CFGNoisePredictor
