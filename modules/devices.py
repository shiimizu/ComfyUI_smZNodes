import sys
import contextlib
import torch
from . import shared
from comfy import model_management

if sys.platform == "darwin":
    from . import mac_specific


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps

cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = model_management.get_torch_device()
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False

def torch_gc():
    model_management.soft_empty_cache()

def get_optimal_device():
    return model_management.get_torch_device()

def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


def randn(seed, shape):
    from modules.shared import opts

    torch.manual_seed(seed)
    if opts.randn_source == "CPU" or device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    from modules.shared import opts

    if opts.randn_source == "CPU" or device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)

def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or model_management.get_torch_device() == torch.device("mps"): # or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    # only cuda
    autocast_device = model_management.get_autocast_device(model_management.get_torch_device())
    # autocast_device = "cuda"
    return torch.autocast(autocast_device)

def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()

class NansException(Exception):
    pass

def test_for_nans(x, where):
    if shared.opts.disable_nan_check:
        return
    if not torch.all(torch.isnan(x)).item():
        return
    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."
        if not shared.opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."
    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."
        if not shared.opts.no_half and not shared.opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."
    message += " Use --disable-nan-check commandline argument to disable this check."
    raise NansException(message)
