
import inspect
from . import errors
from collections import namedtuple

def report_exception(c, job):
    errors.report(f"Error executing callback {job} for {c.script}", exc_info=True)

ScriptCallback = namedtuple("ScriptCallback", ["script", "callback"])
callback_map = dict(
    callbacks_app_started=[],
    callbacks_model_loaded=[],
    callbacks_ui_tabs=[],
    callbacks_ui_train_tabs=[],
    callbacks_ui_settings=[],
    callbacks_before_image_saved=[],
    callbacks_image_saved=[],
    callbacks_cfg_denoiser=[],
    callbacks_cfg_denoised=[],
    callbacks_cfg_after_cfg=[],
    callbacks_before_component=[],
    callbacks_after_component=[],
    callbacks_image_grid=[],
    callbacks_infotext_pasted=[],
    callbacks_script_unloaded=[],
    callbacks_before_ui=[],
    callbacks_on_reload=[],
    callbacks_list_optimizers=[],
    callbacks_list_unets=[],
)

def list_optimizers_callback():
    res = []

    for c in callback_map['callbacks_list_optimizers']:
        try:
            c.callback(res)
        except Exception:
            report_exception(c, 'list_optimizers')

    return res


def on_list_optimizers(callback):
    """register a function to be called when UI is making a list of cross attention optimization options.
    The function will be called with one argument, a list, and shall add objects of type modules.sd_hijack_optimizations.SdOptimization
    to it."""

    add_callback(callback_map['callbacks_list_optimizers'], callback)

def add_callback(callbacks, fun):
    stack = [x for x in inspect.stack() if x.filename != __file__]
    filename = stack[0].filename if stack else 'unknown file'

    callbacks.append(ScriptCallback(filename, fun))
