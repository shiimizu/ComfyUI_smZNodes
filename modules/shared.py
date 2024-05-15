from comfy.model_management import vram_state, VRAMState
import logging
from copy import deepcopy
from comfy.cli_args import args
from comfy import model_management
from . import devices

log = logging.getLogger("sd")
options_templates = {}
loaded_hypernetworks = []
xformers_available = model_management.XFORMERS_IS_AVAILABLE
device = devices.device

class Options: ...

opts = Options()
opts.prompt_attention = 'A1111 parser'
opts.prompt_mean_norm = True
opts.comma_padding_backtrack = 20
opts.CLIP_stop_at_last_layers = 1
opts.enable_emphasis = True
opts.use_old_emphasis_implementation = False
opts.disable_nan_check = True
opts.pad_cond_uncond = False
opts.s_min_uncond = 0.0
opts.upcast_sampling = True
opts.upcast_attn = not getattr(args, 'dont_upcast_attention', False)
opts.textual_inversion_add_hashes_to_infotext  = False
opts.encode_count = 0
opts.max_chunk_count = 0
opts.return_batch_chunks = False
opts.noise = None
opts.start_step = None
opts.pad_with_repeats = True
opts.randn_source = "cpu"
opts.lora_functional = False
opts.use_old_scheduling = True
opts.eta_noise_seed_delta = 0
opts.multi_conditioning = False
opts.eta = 1.0
opts.s_churn = 0.0
opts.s_tmin = 0.0
opts.s_tmax = 0.0 or float('inf')
opts.s_noise = 1.0

opts.use_CFGDenoiser = False
opts.sgm_noise_multiplier = True
opts.debug= False

opts.sdxl_crop_top = 0
opts.sdxl_crop_left = 0
opts.sdxl_refiner_low_aesthetic_score = 2.5
opts.sdxl_refiner_high_aesthetic_score = 6.0

sd_model = Options()
sd_model.cond_stage_model = Options()

cmd_opts = Options()

opts.batch_cond_uncond = False
cmd_opts.lowvram = vram_state == VRAMState.LOW_VRAM
cmd_opts.medvram = vram_state == VRAMState.NORMAL_VRAM
should_batch_cond_uncond = lambda: opts.batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
opts.batch_cond_uncond = True

opts_default = deepcopy(opts)

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