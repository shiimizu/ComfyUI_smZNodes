import torch
import copy
from types import MethodType
from comfy.sd import CLIP
from . import devices
from ..smZNodes import FrozenCLIPEmbedderWithCustomWordsCustom, get_learned_conditioning

class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenize([embedding.name])[0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None

class StableDiffusionModelHijack:
    fixes = None
    comments = []
    layers = None
    circular_enabled = False
    clip = None
    tokenizer = None
    optimization_method = None
    embedding_db = EmbeddingDatabase()
    # embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()

    # def __init__(self):
    #     self.embedding_db.add_embedding_dir(opts.embeddings_dir)

    def hijack(self, m:CLIP):

        if "SD1ClipModel" == type(m.cond_stage_model).__name__:
            self.clip_orig: CLIP = m.clone()
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            model_embeddings.token_embedding.weight = model_embeddings.token_embedding.wrapped._parameters.get('weight')
            m.cond_stage_model = FrozenCLIPEmbedderWithCustomWordsCustom(m.cond_stage_model, self)

            self.cond_stage_model = m.cond_stage_model

            # get_learned_conditioning() -> sd.py's self.cond_stage_model.encode(c) -> forward()
            # The is no `get_learned_conditioning()` so we add it, but make it
            # use our `m.cond_stage_model.forward()` which runs `torch.nn.Module`'s `forward()` function
            # from `FrozenCLIPEmbedderWithCustomWordsBase`
            m.cond_stage_forward = "forward"
            m.cond_stage_model.get_learned_conditioning = MethodType(get_learned_conditioning, m)

            self.clip = m.cond_stage_model
        elif "SDXLClipModel" == type(m.cond_stage_model).__name__:
            self.clip_orig: CLIP = copy.copy(m.clone())
            model_embeddings = m.cond_stage_model.clip_g.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            model_embeddings.token_embedding.weight = model_embeddings.token_embedding.wrapped._parameters.get('weight')
            m.cond_stage_model.clip_g = FrozenCLIPEmbedderWithCustomWordsCustom(m.cond_stage_model.clip_g, self, "clip_g")
            m.cond_stage_model.clip_g.clip_layer = m.cond_stage_model.clip_g.wrapped.clip_layer
            m.cond_stage_model.clip_g.reset_clip_layer = m.cond_stage_model.clip_g.wrapped.reset_clip_layer

            model_embeddings = m.cond_stage_model.clip_l.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            model_embeddings.token_embedding.weight = model_embeddings.token_embedding.wrapped._parameters.get('weight')
            m.cond_stage_model.clip_l = FrozenCLIPEmbedderWithCustomWordsCustom(m.cond_stage_model.clip_l, self, "clip_l")
            m.cond_stage_model.clip_l.clip_layer = m.cond_stage_model.clip_l.wrapped.clip_layer
            m.cond_stage_model.clip_l.reset_clip_layer = m.cond_stage_model.clip_l.wrapped.reset_clip_layer

            self.cond_stage_model = m.cond_stage_model
            # self.cond_stage_model.clip_g = m.cond_stage_model.clip_g
            # self.cond_stage_model.clip_l = m.cond_stage_model.clip_l

            m.cond_stage_forward = "forward"
            m.cond_stage_model.get_learned_conditioning = MethodType(get_learned_conditioning, m)

            self.clip = m.cond_stage_model

        apply_weighted_forward(self.clip)

    def undo_hijack(self, m):
        if "SDXLClipModel" == type(m.cond_stage_model).__name__:
            m.cond_stage_model.clip_g = m.cond_stage_model.clip_g.wrapped
            model_embeddings = m.cond_stage_model.clip_g.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped

            m.cond_stage_model.clip_l = m.cond_stage_model.clip_l.wrapped
            model_embeddings = m.cond_stage_model.clip_l.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        else:
            m.cond_stage_model = m.cond_stage_model.wrapped
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        # undo_optimizations()
        undo_weighted_forward(m)
        self.apply_circular(False)
        # self.layers = None
        self.clip = None
        self.cond_stage_model = None

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
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = devices.cond_cast_unet(embedding.vec)
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
