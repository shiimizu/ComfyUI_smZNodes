import torch
import comfy
import comfy.sd1_clip
from torch.nn.functional import silu
from types import MethodType
import comfy.ldm.modules.diffusionmodules
import comfy.ldm.modules.diffusionmodules.model
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.ldm.modules.attention
from . import devices, shared, sd_hijack_unet, sd_hijack_optimizations, script_callbacks, errors
from .textual_inversion import textual_inversion
from ..smZNodes import FrozenCLIPEmbedderWithCustomWordsCustom, FrozenOpenCLIPEmbedder2WithCustomWordsCustom, get_learned_conditioning
if not hasattr(comfy.ldm.modules.diffusionmodules.model, "nonlinearity_orig"):
    comfy.ldm.modules.diffusionmodules.model.nonlinearity_orig = comfy.ldm.modules.diffusionmodules.model.nonlinearity
if not hasattr(comfy.ldm.modules.diffusionmodules.openaimodel, "th_orig"):
    comfy.ldm.modules.diffusionmodules.openaimodel.th_orig = comfy.ldm.modules.diffusionmodules.openaimodel.th

comfy.ldm.modules.attention.CrossAttention.forward_orig = comfy.ldm.modules.attention.CrossAttention.forward
comfy.ldm.modules.diffusionmodules.model.AttnBlock.forward_orig = comfy.ldm.modules.diffusionmodules.model.AttnBlock.forward

optimizers = []
current_optimizer: sd_hijack_optimizations.SdOptimization = None
already_optimized = False # temp fix for displaying info since two cliptextencode's will run

def list_optimizers():
    script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
    new_optimizers = script_callbacks.list_optimizers_callback()

    new_optimizers = [x for x in new_optimizers if x.is_available()]

    new_optimizers = sorted(new_optimizers, key=lambda x: x.priority, reverse=True)

    optimizers.clear()
    optimizers.extend(new_optimizers)


def apply_optimizations(option=None):
    global already_optimized
    if already_optimized:
        display = False
    list_optimizers()
    global current_optimizer

    undo_optimizations()

    if len(optimizers) == 0:
        # a script can access the model very early, and optimizations would not be filled by then
        current_optimizer = None
        return ''

    comfy.ldm.modules.diffusionmodules.model.nonlinearity = silu
    comfy.ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    # sgm.modules.diffusionmodules.model.nonlinearity = silu
    # sgm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    if current_optimizer is not None:
        current_optimizer.undo()
        current_optimizer = None

    selection = option or shared.opts.cross_attention_optimization
    if selection == "Automatic" and len(optimizers) > 0:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt and getattr(shared.cmd_opts, x.cmd_opt, False)]), optimizers[0])
    else:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt == selection]), None)
    if selection == "None":
        matching_optimizer = None
    elif selection == "Automatic" and shared.cmd_opts.disable_opt_split_attention:
        matching_optimizer = None
    elif matching_optimizer is None:
        matching_optimizer = optimizers[0]

    if matching_optimizer is not None:
        if shared.opts.debug:
            print(f"Applying attention optimization: {matching_optimizer.name}... ", end='')
        matching_optimizer.apply()
        already_optimized = True
        if shared.opts.debug:
            print("done.")
        current_optimizer = matching_optimizer
        return current_optimizer
    else:
        # if shared.opts.debug:
            # print("Disabling attention optimization")
        return ''

def undo_optimizations():
    sd_hijack_optimizations.undo()
    comfy.ldm.modules.diffusionmodules.model.nonlinearity = comfy.ldm.modules.diffusionmodules.model.nonlinearity_orig
    comfy.ldm.modules.diffusionmodules.openaimodel.th = comfy.ldm.modules.diffusionmodules.openaimodel.th_orig

class StableDiffusionModelHijack:
    fixes = None
    comments = []
    layers = None
    circular_enabled = False
    clip = None
    tokenizer = None
    optimization_method = None
    embedding_db = textual_inversion.EmbeddingDatabase()

    def apply_optimizations(self, option=None):
        try:
            self.optimization_method = apply_optimizations(option)
        except Exception as e:
            errors.display(e, "applying optimizations")
            undo_optimizations()

    def hijack(self, m: comfy.sd1_clip.SD1ClipModel):
        tokenizer_parent = m.tokenizer # SD1Tokenizer
        # SDTokenizer
        tokenizer_parent2 = getattr(tokenizer_parent, tokenizer_parent.clip) if hasattr(tokenizer_parent, 'clip') else tokenizer_parent
        tokenizer = getattr(tokenizer_parent, tokenizer_parent.clip).tokenizer if hasattr(tokenizer_parent, 'clip') else tokenizer_parent.tokenizer
        if hasattr(m, 'clip'):
            m = getattr(m, m.clip)
        model_embeddings = m.transformer.text_model.embeddings
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self, "clip_g" if "SDXLClipG" in type(m).__name__ else "clip_l")
        model_embeddings.token_embedding.weight = model_embeddings.token_embedding.wrapped._parameters.get('weight').to(device=devices.device)
        m.tokenizer_parent0 = tokenizer_parent
        m.tokenizer_parent = tokenizer_parent2
        m.tokenizer = tokenizer
        m = FrozenOpenCLIPEmbedder2WithCustomWordsCustom(m, self) if "SDXLClipG" in type(m).__name__ else FrozenCLIPEmbedderWithCustomWordsCustom(m, self)
        m.clip_layer = getattr(m.wrapped, "clip_layer", None)
        m.reset_clip_layer = getattr(m.wrapped, "reset_clip_layer", None)
        m.set_clip_options = getattr(m.wrapped, "set_clip_options", None)
        m.reset_clip_options = getattr(m.wrapped, "reset_clip_options", None)
        m.transformer = getattr(m.wrapped, "transformer", None)
        self.cond_stage_model = m
        self.clip = m

        apply_weighted_forward(self.clip)
        self.apply_optimizations()

    def undo_hijack(self, m):
        try:
            m = m.wrapped
            model_embeddings = m.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
            undo_optimizations()
            undo_weighted_forward(m)
            self.apply_circular(False)
            # self.layers = None
            self.clip = None
            self.cond_stage_model = None
        except Exception as err:
            print(err)

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return

        self.circular_enabled = enable

        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []

    def get_prompt_lengths(self, text):
        if self.clip is None:
            return 0, 0
        _, token_count = self.clip.process_texts([text])
        return token_count, self.clip.get_target_prompt_token_count(token_count)

model_hijack = StableDiffusionModelHijack()

def weighted_loss(sd_model, pred, target, mean=True):
    #Calculate the weight normally, but ignore the mean
    loss = sd_model._old_get_loss(pred, target, mean=False) # pylint: disable=protected-access

    #Check if we have weights available
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    #Return the loss, as mean if specified
    return loss.mean() if mean else loss

def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        #Temporarily append weights to a place accessible during loss calc
        sd_model._custom_loss_weight = w # pylint: disable=protected-access

        #Replace 'get_loss' with a weight-aware one. Otherwise we need to reimplement 'forward' completely
        #Keep 'get_loss', but don't overwrite the previous old_get_loss if it's already set
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss # pylint: disable=protected-access
        sd_model.get_loss = MethodType(weighted_loss, sd_model)

        #Run the standard forward function, but with the patched 'get_loss'
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            #Delete temporary weights if appended
            del sd_model._custom_loss_weight
        except AttributeError:
            pass

        #If we have an old loss function, reset the loss function to the original one
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss # pylint: disable=protected-access
            del sd_model._old_get_loss

def apply_weighted_forward(sd_model):
    #Add new function 'weighted_forward' that can be called to calc weighted loss
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)

def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        try:
            inputs_embeds = self.wrapped(input_ids)
        except Exception:
            inputs_embeds = self.wrapped(input_ids.cpu())

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                vec = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = devices.cond_cast_unet(vec)
                if emb.device != tensor.device:
                    emb = emb.to(device=tensor.device)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                try:
                    tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])
                except Exception as err:
                    print("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored", tensor.shape[0], emb.shape[1])
                    # raise err
            vecs.append(tensor)

        return torch.stack(vecs)
