import torch
import numpy as np
from comfy.model_patcher import ModelPatcher
from . import shared, rng_philox

def randn_without_seed(x, generator=None, randn_source="cpu"):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""
    if randn_source == "nv":
        return torch.asarray(generator.randn(x.size()), device=x.device)
    else:
        if generator is not None and generator.device.type == "cpu":
            return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device='cpu', generator=generator).to(device=x.device)
        else:
            return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, generator, randn_source):
        # self.rng = p.rng
        self.generator = generator
        self.randn_source = randn_source

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return randn_without_seed(x, generator=self.generator, randn_source=self.randn_source)

def _find_outer_instance(target:str, target_type=None, callback=None):
    import inspect
    frame = inspect.currentframe()
    i = 0
    while frame and i < 10:
        if target in frame.f_locals:
            if callback is not None:
                return callback(frame)
            else:
                found = frame.f_locals[target]
                if isinstance(found, target_type):
                    return found
        frame = frame.f_back
        i += 1
    return None

def prepare_noise(latent_image, seed, noise_inds=None, device='cpu'):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    opts = None
    model = _find_outer_instance('model', ModelPatcher)
    if (model is not None and (opts:=model.model_options.get(shared.Options.KEY, None)) is None) or opts is None:
        import comfy.samplers
        guider = _find_outer_instance('guider', comfy.samplers.CFGGuider)
        model = getattr(guider, 'model_patcher', None)
    if (model is not None and (opts:=model.model_options.get(shared.Options.KEY, None)) is None) or opts is None:
        opts = shared.opts_default.clone()
        device = 'cpu'

    if opts.randn_source == 'gpu':
        import comfy.model_management
        device = comfy.model_management.get_torch_device()

    def get_generator(seed):
        nonlocal device
        nonlocal opts
        _generator = torch.Generator(device=device)
        generator = _generator.manual_seed(seed)
        if opts.randn_source == 'nv':
            generator = rng_philox.Generator(seed)
        return generator
    generator = generator_eta = get_generator(seed)

    if opts.eta_noise_seed_delta > 0:
        seed = min(int(seed + opts.eta_noise_seed_delta), int(0xffffffffffffffff))
        generator_eta = get_generator(seed)

    # hijack randn_like
    import comfy.k_diffusion.sampling
    comfy.k_diffusion.sampling.torch = TorchHijack(generator_eta, opts.randn_source)

    if noise_inds is None:
        shape = latent_image.size()
        if opts.randn_source == 'nv':
            return torch.asarray(generator.randn(shape), device='cpu')
        else:
            return torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        shape = [1] + list(latent_image.size())[1:]
        if opts.randn_source == 'nv':
            noise = torch.asarray(generator.randn(shape), device='cpu')
        else:
            noise = torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

def hook_prepare_noise():
    import comfy.sample
    comfy.sample.prepare_noise = prepare_noise