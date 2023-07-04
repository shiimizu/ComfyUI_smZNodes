import torch
from types import MethodType
from .sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords
from comfy.sd import CLIP

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

        # ids = model.cond_stage_model.tokenize([embedding.name])[0]
        ids = model.tokenizer([embedding.name])["input_ids"][0][1:-1]

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

    def hijack(self, m):
        self.clip_orig: CLIP = m.clone()
        m.cond_stage_model.tokenizer = m.tokenizer

        backup_embeds = m.cond_stage_model.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        dtype = backup_embeds.weight.dtype

        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        model_embeddings.token_embedding.weight = backup_embeds.weight
        m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)
        m.cond_stage_model.device = device
        model_embeddings.token_embedding.device = device
        model_embeddings.token_embedding.dtype = dtype

        self.clip = m.cond_stage_model

        # self.clip.device = self.clip.wrapped.device
        apply_weighted_forward(self.clip)

    def undo_hijack(self, m):
        m.cond_stage_model = m.cond_stage_model.wrapped
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
        if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
            model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        # undo_optimizations()
        undo_weighted_forward(m)
        self.apply_circular(False)
        # self.layers = None
        self.clip = None

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
# class EmbeddingsWithFixes(torch.nn.Embedding):

    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.weight : torch.Tensor= None

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
                        
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                # emb = devices.cond_cast_unet(embedding.vec) # sets the dtype like below
                emb = embedding.vec.to(self.dtype, device=tensor.device)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])
            vecs.append(tensor)

        return torch.stack(vecs)
