import torch
from collections import namedtuple
from . import prompt_parser, emphasis
from comfy import model_management


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
        self._tokenizer = tokenizer

        self.text_encoder = text_encoder

        self.emphasis = emphasis.get_current_option(emphasis_name)()
        self.min_length = self.min_length or self.max_length
        self.id_end = self.end_token
        self.id_pad = self.pad_token
        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab.get(',</w>', None)
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
        if hasattr(self.tokenizer, w): delattr(self.tokenizer, w)
        if hasattr(self._tokenizer, w): delattr(self._tokenizer, w)

    def tokenize_with_weights(self, texts, return_word_ids=False):
        tokens_and_weights = []
        cache = {}
        for line in texts:
            if line not in cache:
                chunks, token_count = self.tokenize_line(line)
                line_tokens_and_weights = []

                # Pad all chunks to the length of the longest chunk
                max_tokens = 0
                for chunk in chunks:
                    max_tokens = max (len(chunk.tokens), max_tokens)

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers
                    remaining_count = max_tokens - len(tokens)
                    if remaining_count > 0:
                        tokens += [self.id_pad] * remaining_count
                        multipliers += [1.0] * remaining_count
                    line_tokens_and_weights.append((tokens, multipliers))
                cache[line] = line_tokens_and_weights

            tokens_and_weights.extend(cache[line])
        return tokens_and_weights

    def encode_token_weights(self, token_weight_pairs):
        if isinstance(token_weight_pairs[0], str):
            token_weight_pairs = self.tokenize_with_weights(token_weight_pairs)
        elif isinstance(token_weight_pairs[0], list):
            token_weight_pairs = list(map(lambda x: (list(map(lambda y: y[0], x)), list(map(lambda y: y[1], x))), token_weight_pairs))

        target_device = model_management.text_encoder_offload_device()
        zs = []
        cache = {}
        for tokens, multipliers in token_weight_pairs:
            token_key = (tuple(tokens), tuple(multipliers))
            if token_key not in cache:
                z = self.process_tokens([tokens], [multipliers])[0]
                cache[token_key] = z
            zs.append(cache[token_key])
        return torch.stack(zs).to(target_device), None

    def __call__(self, texts):
        tokens = self.tokenize_with_weights(texts)
        return self.encode_token_weights(tokens)

    def process_tokens(self, batch_tokens, batch_multipliers):
        tokens = torch.asarray(batch_tokens)

        z = self.encode_with_transformers(tokens)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
