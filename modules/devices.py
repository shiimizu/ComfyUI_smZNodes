import torch
device=None
dtype_unet = torch.float16
unet_needs_upcast = False

def cond_cast_unet(tensor: torch.Tensor):
    return tensor.to(dtype=dtype_unet) if unet_needs_upcast else tensor
