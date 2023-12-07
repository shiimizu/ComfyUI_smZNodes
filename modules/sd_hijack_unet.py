import torch
from packaging import version

from . import devices
from .sd_hijack_utils import CondFunc
from torch.nn.functional import silu
import comfy
from comfy import ldm
import contextlib

class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)

th = TorchHijackForUnet()

from . import sd_hijack_optimizations
from comfy.model_base import BaseModel
from functools import wraps
sdp_no_mem = sd_hijack_optimizations.SdOptimizationSdpNoMem()
BaseModel.apply_model_orig = BaseModel.apply_model

# @contextmanager
class ApplyOptimizationsContext:
    def __init__(self):
        self.nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
        self.th = ldm.modules.diffusionmodules.openaimodel.th
        ldm.modules.diffusionmodules.model.nonlinearity = silu
        ldm.modules.diffusionmodules.openaimodel.th = th
        sdp_no_mem.apply()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ldm.modules.diffusionmodules.model.nonlinearity = self.nonlinearity
        ldm.modules.diffusionmodules.openaimodel.th = self.th
        sd_hijack_optimizations.undo()



def ApplyOptimizationsContext3(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with ApplyOptimizationsContext():
            return func(*args, **kwargs)
    return wrapper

precision_scope_null = lambda a, dtype=None: contextlib.nullcontext(a)

# def apply_model(orig_func, self, x_noisy, t, c_concat=None, c_crossattn=None, c_adm=None, control=None, transformer_options={}, *args, **kwargs):
def apply_model(orig_func, self, *args, **kwargs):
    transformer_options = kwargs['transformer_options'] if 'transformer_options' in kwargs else {}
    c_crossattn = kwargs['c_crossattn'] if 'c_crossattn' in kwargs else args[3]
    x_noisy = kwargs['x_noisy'] if 'x_noisy' in kwargs else args[0]
    if not transformer_options.get('from_smZ', False):
        return self.apply_model_orig(*args, **kwargs)

    cond=c_crossattn
    if isinstance(cond, dict):
        for y in cond.keys():
            if isinstance(cond[y], list):
                cond[y] = [x.to(devices.dtype_unet) if isinstance(x, torch.Tensor) else x for x in cond[y]]
            else:
                cond[y] = cond[y].to(devices.dtype_unet) if isinstance(cond[y], torch.Tensor) else cond[y]

    if x_noisy.dtype != torch.float32:
        precision_scope = torch.autocast
    else:
        precision_scope = precision_scope_null

    with precision_scope(comfy.model_management.get_autocast_device(x_noisy.device), dtype=x_noisy.dtype): # , torch.float32):
    # with devices.autocast():
        out = orig_func(self, *args, **kwargs).float()
    return out

class GELUHijack(torch.nn.GELU, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.GELU.__init__(self, *args, **kwargs)
    def forward(self, x):
        if devices.unet_needs_upcast:
            return torch.nn.GELU.forward(self.float(), x.float()).to(devices.dtype_unet)
        else:
            return torch.nn.GELU.forward(self, x)

ddpm_edit_hijack = None
def hijack_ddpm_edit():
    global ddpm_edit_hijack
    if not ddpm_edit_hijack:
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
        ddpm_edit_hijack = CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)


unet_needs_upcast = lambda *args, **kwargs: devices.unet_needs_upcast
# CondFunc('comfy.model_base.BaseModel.apply_model', apply_model, unet_needs_upcast)
# CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)
# CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', lambda orig_func, timesteps, *args, **kwargs: orig_func(timesteps, *args, **kwargs).to(torch.float32 if timesteps.dtype == torch.int64 else devices.dtype_unet), unet_needs_upcast)
# if version.parse(torch.__version__) <= version.parse("1.13.2") or torch.cuda.is_available():
#     CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', lambda orig_func, self, *args, **kwargs: orig_func(self.float(), *args, **kwargs), unet_needs_upcast)
#     CondFunc('ldm.modules.attention.GEGLU.forward', lambda orig_func, self, x: orig_func(self.float(), x.float()).to(devices.dtype_unet), unet_needs_upcast)
#     try:
#         CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)
#     except:
#         CondFunc('comfy.t2i_adapter.adapter.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)
first_stage_cond = lambda _, self, *args, **kwargs: devices.unet_needs_upcast and self.model.diffusion_model.dtype == torch.float16
first_stage_sub = lambda orig_func, self, x, **kwargs: orig_func(self, x.to(devices.dtype_vae), **kwargs)
# CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
# CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
# CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).float(), first_stage_cond)

# CondFunc('sgm.modules.diffusionmodules.wrappers.OpenAIWrapper.forward', apply_model, unet_needs_upcast)
# CondFunc('sgm.modules.diffusionmodules.openaimodel.timestep_embedding', lambda orig_func, timesteps, *args, **kwargs: orig_func(timesteps, *args, **kwargs).to(torch.float32 if timesteps.dtype == torch.int64 else devices.dtype_unet), unet_needs_upcast)
