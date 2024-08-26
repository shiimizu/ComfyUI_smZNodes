import torch
import inspect
from collections import namedtuple
from . import parsing, emphasis
from comfy import model_management
from ..modules import prompt_parser


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])

def populate_self_variables(self, from_):
    attrs_from = vars(from_)
    attrs_self = vars(self)
    attrs_self.update(attrs_from)

class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class T5TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, emphasis_name="Original", min_length=256):
        super().__init__()
        populate_self_variables(self, tokenizer)

        self.text_encoder = text_encoder

        self.emphasis = emphasis.get_current_option(emphasis_name)()
        self.min_length = self.min_length or self.max_length
        self.id_end = self.end_token
        self.id_pad = self.pad_token
        self.tokenizer._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
        

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def encode_with_transformers(self, tokens):
        try:
            z, pooled = self.text_encoder(tokens)
        except Exception:
            z, pooled = self.text_encoder(tokens.tolist())
        return z

    def tokenize_line(self, line):
        parsed = prompt_parser.parse_prompt_attention(line)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            chunk.tokens = chunk.tokens + [self.id_end]
            chunk.multipliers = chunk.multipliers + [1.0]
            current_chunk_length = len(chunk.tokens)

            token_count += current_chunk_length
            remaining_count = self.min_length - current_chunk_length

            if remaining_count > 0:
                chunk.tokens += [self.id_pad] * remaining_count
                chunk.multipliers += [1.0] * remaining_count

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count
 
    def unhook(self):
        w = '_eventual_warn_about_too_long_sequence'
        if hasattr(self.tokenizer, w):
            delattr(self.tokenizer, w)

    def tokenize_with_weights(self, texts, return_word_ids=False):
        return texts

    def encode_token_weights(self, token_weight_pairs):
        target_device = model_management.text_encoder_device()
        z = self(token_weight_pairs).to(target_device)
        return z, None

    def __call__(self, texts):
        target_device = model_management.text_encoder_device()
        zs = []
        cache = {}
        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks, token_count = self.tokenize_line(line)
                line_z_values = []
                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers
                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)
        return torch.stack(zs).to(target_device)

    def process_tokens(self, batch_tokens, batch_multipliers):
        tokens = torch.asarray(batch_tokens)

        z = self.encode_with_transformers(tokens)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
