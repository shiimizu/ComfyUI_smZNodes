import torch
from . import devices
from . import prompt_parser
from . import shared
from comfy import model_management

# This is a `model` that gets fed into a sampler who then calls it which calls
# torch's forward function to return a denoise
class CFGDenoiser(torch.nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.step = 0
        self.image_cfg_scale = None
        self.device = model_management.get_torch_device()
        self.sigma_to_t = self.inner_model.sigma_to_t
        self.t_to_sigma = self.inner_model.t_to_sigma
        self.sigmas = self.inner_model.sigmas


    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)

        return denoised

    # def apply_model(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, model_options={}, seed=None):
    #     out = self(x, timestep, uncond, cond, cond_scale, cond_concat, model_options=model_options, seed=seed)
    #     return out

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        model_management.throw_exception_if_processing_interrupted()
        # if state.interrupted or state.skipped:
        #     raise sd_samplers_common.InterruptedException
        # if state.paused:
        #     shared.log.debug('Sampling paused')
        #     while state.paused:
        #         if state.interrupted or state.skipped:
        #             raise sd_samplers_common.InterruptedException
        #         import time
        #         time.sleep(0.1)

        # at self.image_cfg_scale == 1.0 produced results for edit model are the same as with normal sampling,
        # so is_edit_model is set to False to support AND composition.
        # is_edit_model = (shared.sd_model is not None) and hasattr(shared.sd_model, 'cond_stage_key') and (shared.sd_model.cond_stage_key == "edit") and (self.image_cfg_scale is not None) and (self.image_cfg_scale != 1.0)
        is_edit_model = False

        # conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        # uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)
        tensor = cond
        conds_list=[[(0, 1.0)]]

        assert not is_edit_model or all(len(conds) == 1 for conds in conds_list), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        # if shared.sd_model.model.conditioning_key == "crossattn-adm":
        #     image_uncond = torch.zeros_like(image_cond)
        #     make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": c_crossattn, "c_adm": c_adm} # pylint: disable=C3001
        # else:
        #     image_uncond = image_cond
        #     make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": c_crossattn, "c_concat": [c_concat]} # pylint: disable=C3001

        image_uncond = image_cond
        # make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": c_crossattn, "c_concat": [c_concat.to(device=self.device)]} # pylint: disable=C3001
        # make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [ctn.to(device=self.device) for ctn in c_crossattn] if type(c_crossattn) is list else c_crossattn.to(device=self.device),  "c_concat": [c_concat.to(device=self.device)]} # pylint: disable=C3001
        make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [ctn.to(device=self.device) for ctn in c_crossattn] if type(c_crossattn) is list else c_crossattn.to(device=self.device)} # pylint: disable=C3001


        if not is_edit_model:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond])
        else:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond] + [torch.zeros_like(self.init_latent)])

        # denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps, tensor, uncond)
        # cfg_denoiser_callback(denoiser_params)
        # x_in = denoiser_params.x
        # image_cond_in = denoiser_params.image_cond
        # sigma_in = denoiser_params.sigma
        # tensor = denoiser_params.text_cond
        # uncond = denoiser_params.text_uncond
        skip_uncond = False

        # alternating uncond allows for higher thresholds without the quality loss normally expected from raising it
        if self.step % 2 and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            skip_uncond = True
            x_in = x_in[:-batch_size]
            sigma_in = sigma_in[:-batch_size]

        if tensor.shape[1] == uncond.shape[1] or skip_uncond:
            if is_edit_model:
                cond_in = torch.cat([tensor, uncond, uncond])
            elif skip_uncond:
                cond_in = tensor
            else:
                cond_in = torch.cat([tensor, uncond])

            if shared.batch_cond_uncond:
                x_out = self.inner_model(x_in, sigma_in, **make_condition_dict([cond_in], image_cond_in))
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], **make_condition_dict([cond_in[a:b]], image_cond_in[a:b]))
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size*2 if shared.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])

                if not is_edit_model:
                    c_crossattn = [tensor[a:b]]
                else:
                    c_crossattn = torch.cat([tensor[a:b]], uncond)

                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], **make_condition_dict(c_crossattn, image_cond_in[a:b]))

            if not skip_uncond:
                x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], **make_condition_dict([uncond], image_cond_in[-uncond.shape[0]:]))

        denoised_image_indexes = [x[0][0] for x in conds_list]
        if skip_uncond:
            fake_uncond = torch.cat([x_out[i:i+1] for i in denoised_image_indexes])
            x_out = torch.cat([x_out, fake_uncond])  # we skipped uncond denoising, so we put cond-denoised image to where the uncond-denoised image should be

        # denoised_params = CFGDenoisedParams(x_out, state.sampling_step, state.sampling_steps, self.inner_model)
        # cfg_denoised_callback(denoised_params)

        devices.test_for_nans(x_out, "unet")

        # if opts.live_preview_content == "Prompt":
        #     sd_samplers_common.store_latent(torch.cat([x_out[i:i+1] for i in denoised_image_indexes]))
        # elif opts.live_preview_content == "Negative prompt":
        #     sd_samplers_common.store_latent(x_out[-uncond.shape[0]:])

        if is_edit_model:
            denoised = self.combine_denoised_for_edit_model(x_out, cond_scale)
        elif skip_uncond:
            denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
        else:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        # after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
        # cfg_after_cfg_callback(after_cfg_callback_params)
        # denoised = after_cfg_callback_params.x

        self.step += 1
        del x_out
        return denoised
