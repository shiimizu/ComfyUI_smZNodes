import math
import torch
import logging
from collections import namedtuple
from comfy import model_management
from . import emphasis, prompt_parser
from .textual_inversion import EmbeddingDatabase, parse_and_register_embeddings


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
last_extra_generation_params = {}

def populate_self_variables(self, from_):
    attrs_from = vars(from_)
    attrs_self = vars(self)
    attrs_self.update(attrs_from)

class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class CLIPEmbeddingForTextualInversion(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key
        self.weight = self.wrapped.weight

    def forward(self, input_ids, out_dtype):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids, out_dtype)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = emb.to(inputs_embeds)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                try:
                    tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]]).to(dtype=inputs_embeds.dtype)
                except Exception:
                    logging.warning("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {} {} {} '{}'".format(tensor.shape[0], emb.shape[1], self.current_embeds.weight.shape[1], self.textual_inversion_key, embedding.name))

            vecs.append(tensor)

        return torch.stack(vecs)


class ClassicTextProcessingEngine:
    def __init__(
            self, text_encoder, tokenizer, chunk_length=75,
            embedding_dir=None, embedding_key='clip_l', embedding_expected_shape=768, emphasis_name="Original",
            text_projection=False, minimal_clip_skip=1, clip_skip=1, return_pooled=True, final_layer_norm=True
    ):
        super().__init__()
        populate_self_variables(self, tokenizer)
        self._tokenizer = tokenizer

        self.embeddings = EmbeddingDatabase(self.tokenizer, embedding_expected_shape)

        self.text_encoder = text_encoder
        self._try_get_embedding = tokenizer._try_get_embedding

        self.emphasis = emphasis.get_current_option(emphasis_name)()

        self.text_projection = text_projection
        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        self.chunk_length = chunk_length

        self.id_start = self.start_token
        self.id_end = self.end_token
        self.id_pad = self.pad_token

        model_embeddings = text_encoder.transformer.text_model.embeddings
        backup_embeds = self.text_encoder.transformer.get_input_embeddings()
        model_embeddings.token_embedding = CLIPEmbeddingForTextualInversion(model_embeddings.token_embedding, self.embeddings, textual_inversion_key=self.embedding_key)
        model_embeddings.token_embedding.current_embeds = backup_embeds

        vocab = self.tokenizer.get_vocab()
        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1
            if mult != 1.0:
                self.token_mults[ident] = mult

        self.comma_token = vocab.get(',</w>', None)
        self.tokenizer._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
    
    def unhook(self):
        self.text_encoder.transformer.text_model.embeddings.token_embedding = self.text_encoder.transformer.text_model.embeddings.token_embedding.wrapped
        del self._try_get_embedding
        w = '_eventual_warn_about_too_long_sequence'
        if hasattr(self.tokenizer, w): delattr(self.tokenizer, w)
        if hasattr(self._tokenizer, w): delattr(self._tokenizer, w)

    def empty_chunk(self):
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized
    
    def tokenize_with_weights(self, texts, return_word_ids=False):
        texts = [parse_and_register_embeddings(self, text) for text in texts]
        if self.opts.use_old_emphasis_implementation:
            return self.process_texts_past(texts)
        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.embeddings.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.embeddings.fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = (tokens, multipliers)
            zs.append(z)

        return zs

    def encode_token_weights(self, token_weight_pairs):
        if isinstance(token_weight_pairs[0], str):
            token_weight_pairs = self.tokenize_with_weights(token_weight_pairs)
        elif isinstance(token_weight_pairs[0], list):
            token_weight_pairs = list(map(lambda x: ([list(map(lambda y: y[0], x))], [list(map(lambda y: y[1], x))]), token_weight_pairs))

        target_device = model_management.text_encoder_offload_device()
        zs = []
        for tokens, multipliers in token_weight_pairs:
            z = self.process_tokens(tokens, multipliers)
            zs.append(z)
        if self.return_pooled:
            return torch.hstack(zs).to(target_device), zs[0].pooled.to(target_device) if zs[0].pooled is not None else None
        else:
            return torch.hstack(zs).to(target_device)

    def encode_with_transformers(self, tokens):
        try:
            z, pooled = self.text_encoder(tokens)
        except Exception:
            z, pooled = self.text_encoder(tokens.tolist())
        z.pooled = pooled
        return z

    def tokenize_line(self, line):
        parsed = prompt_parser.parse_prompt_attention(line)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                comma_padding_backtrack = 20

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.embeddings.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vectors)
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def __call__(self, texts):
        tokens = self.tokenize_with_weights(texts)
        return self.encode_token_weights(tokens)

    def process_tokens(self, remade_batch_tokens, batch_multipliers, *args, **kwargs):
        try:
            tokens = torch.asarray(remade_batch_tokens)

            if self.id_end != self.id_pad:
                for batch_pos in range(len(remade_batch_tokens)):
                    index = remade_batch_tokens[batch_pos].index(self.id_end)
                    tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad

            z = self.encode_with_transformers(tokens)
        except ValueError:
            # Tokens including textual inversion embeddings in the list.
            # i.e. tensors in the list along with tokens.
            z = self.encode_with_transformers(remade_batch_tokens)

        pooled = getattr(z, 'pooled', None)

        self.emphasis.tokens = remade_batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        if pooled is not None:
            z.pooled = pooled

        return z
