from comfy.model_management import vram_state, VRAMState
import logging

log = logging.getLogger("sd")
options_templates = {}

class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value): # pylint: disable=inconsistent-return-statements
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                # if cmd_opts.freeze:
                #     log.warning(f'Settings are frozen: {key}')
                #     return
                # if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                #     log.warning(f'Settings key is restricted: {key}')
                #     return
                self.data[key] = value
                return
        return super(Options, self).__setattr__(key, value) # pylint: disable=super-with-arguments

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super(Options, self).__getattribute__(item) # pylint: disable=super-with-arguments


opts = Options()
opts.data['prompt_attention'] = 'A1111 parser'
opts.data['comma_padding_backtrack'] = None
opts.data['prompt_mean_norm'] = True
opts.data["comma_padding_backtrack"] = 20
opts.data["clip_skip"] = None
opts.data['enable_emphasis'] = True
opts.data['use_old_emphasis_implementation'] = False
opts.data['disable_nan_check'] = True
opts.data['always_batch_cond_uncond'] = False

opts.data['use_CFGDenoiser'] = False
opts.data['disable_max_denoise'] = False

# batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
batch_cond_uncond = opts.always_batch_cond_uncond or not (vram_state == VRAMState.LOW_VRAM or vram_state == VRAMState.NORMAL_VRAM)
