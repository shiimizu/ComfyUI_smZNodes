from comfy.model_management import vram_state, VRAMState
import logging
import sys
from comfy.cli_args import args
from comfy import model_management
from . import devices

log = logging.getLogger("sd")
options_templates = {}
loaded_hypernetworks = []
xformers_available = model_management.XFORMERS_IS_AVAILABLE
device = devices.device

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
opts.data['prompt_mean_norm'] = True
opts.data["comma_padding_backtrack"] = 20
opts.data["CLIP_stop_at_last_layers"] = 1
opts.data['enable_emphasis'] = True
opts.data['use_old_emphasis_implementation'] = False
opts.data['disable_nan_check'] = True
opts.data['pad_cond_uncond'] = False
opts.data['upcast_sampling'] = False
opts.data['upcast_attn'] = not args.dont_upcast_attention
opts.data['textual_inversion_add_hashes_to_infotext']  = False
opts.data['encode_count'] = 0
opts.data['max_chunk_count'] = 0
opts.data['return_batch_chunks'] = False


opts.data['use_CFGDenoiser'] = True
opts.data['disable_max_denoise'] = False


opts.data['sdxl_crop_top'] = 0
opts.data['sdxl_crop_left'] = 0
opts.data['sdxl_refiner_low_aesthetic_score'] = 2.5
opts.data['sdxl_refiner_high_aesthetic_score'] = 6.0

sd_model = Options()
sd_model.cond_stage_model = Options()

cmd_opts = Options()

cmd_opts.always_batch_cond_uncond = False
cmd_opts.lowvram = vram_state == VRAMState.LOW_VRAM
cmd_opts.medvram = vram_state == VRAMState.NORMAL_VRAM
should_batch_cond_uncond = lambda: cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
batch_cond_uncond = should_batch_cond_uncond()

cmd_opts.xformers = xformers_available
cmd_opts.force_enable_xformers = xformers_available

opts.cross_attention_optimization = "None"
# opts.cross_attention_optimization = "opt_sdp_no_mem_attention"
# opts.cross_attention_optimization = "opt_sub_quad_attention"
cmd_opts.sub_quad_q_chunk_size = 512
cmd_opts.sub_quad_kv_chunk_size = 512
cmd_opts.sub_quad_chunk_threshold = 80
cmd_opts.token_merging_ratio = 0.0
cmd_opts.token_merging_ratio_img2img = 0.0
cmd_opts.token_merging_ratio_hr = 0.0
cmd_opts.sd_vae_sliced_encode = False
cmd_opts.disable_opt_split_attention = False