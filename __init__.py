import os
import shutil
import importlib
import subprocess
from pathlib import Path

def init_modules() :
    from sys import modules
    return set(modules.values())

def reload_modules(m) :
    from sys import modules
    for module in set(modules.values()) - m :
        try:
            importlib.reload(module)
        except Exception as e: ...

PRELOADED_MODULES = init_modules()

def install(module):
    if importlib.util.find_spec(module) is not None: return
    import sys
    try:
        print(f"\033[92m[smZNodes] \033[0;31m{module} is not installed. Attempting to install...\033[0m")
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
        reload_modules(PRELOADED_MODULES)
        print(f"\033[92m[smZNodes] {module} Installed!\033[0m")
    except Exception as e:
        print(f"\033[92m[smZNodes] \033[0;31mFailed to install {module}.\033[0m")

for pkg in ['compel', 'lark']:
    install(pkg)

# ============================
# web

cwd_path = Path(__file__).parent
comfy_path = cwd_path.parent.parent

def setup_web_extension():
    import nodes
    web_extension_path = os.path.join(comfy_path, "web", "extensions", "smZNodes")
    if os.path.exists(web_extension_path):
        shutil.rmtree(web_extension_path)
    if not hasattr(nodes, "EXTENSION_WEB_DIRS"):
        if not os.path.exists(web_extension_path):
            os.makedirs(web_extension_path)
        js_src_path = os.path.join(cwd_path, "web/js", "smZdynamicWidgets.js")
        shutil.copy(js_src_path, web_extension_path)

setup_web_extension()

# ============================

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

from .smZNodes import add_custom_samplers, register_hooks

add_custom_samplers()
register_hooks()