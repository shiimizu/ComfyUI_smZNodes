import torch
from comfy.sd import CLIP
from comfy.sd1_clip import SD1ClipModel
from typing import List, Tuple
from types import MethodType
from .modules import prompt_parser
from .modules import shared, devices
from .modules.shared import opts
from .modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords
from comfy.sd1_clip import SD1Tokenizer, SD1ClipModel, unescape_important, escape_important
from comfy.ldm.modules.distributions.distributions import DiagonalGaussianDistribution


def encode_from_tokens_with_custom_mean(clip: CLIP, tokens, return_pooled=False):
    '''
    The function is our rendition of `clip.encode_from_tokens()`.
    It still calls `clip.encode_from_tokens()` but hijacks the
    `clip.cond_stage_model.encode_token_weights()` method
    so we can run our own version of `encode_token_weights()`

    Originally from `sd.py`: `encode_from_tokens()`
    '''
    ret = None
    encode_token_weights_backup = clip.cond_stage_model.encode_token_weights
    try:
        clip.cond_stage_model.encode_token_weights = MethodType(encode_token_weights_customized, clip.cond_stage_model)
        ret = clip.encode_from_tokens(tokens, return_pooled)
        clip.cond_stage_model.encode_token_weights = encode_token_weights_backup
    except Exception as error:
        clip.cond_stage_model.encode_token_weights = encode_token_weights_backup
        raise error

    return ret

def encode_token_weights_customized(self: SD1ClipModel, token_weight_pairs):
    to_encode = list(self.empty_tokens)
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        weights = list(map(lambda a: a[1], x))
        to_encode.append(tokens)

    out, pooled = self.encode(to_encode)
    zw = torch.asarray(weights).to(device=out.device)

    z_empty = out[0:1]
    if pooled.shape[0] > 1:
        first_pooled = pooled[1:2]
    else:
        first_pooled = pooled[0:1]

    output = []
    for i in range(1, out.shape[0]):
        # 3D -> 2D
        z = out[i:i+1]
        batch_multipliers = zw[i:i+1]
        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        if opts.prompt_mean_norm:
            original_mean = z.mean()
            z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
            new_mean = z.mean()
            z = z * (original_mean / new_mean)
        else:
            z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
            # for i in range(len(z)):
            #     for j in range(len(z[i])):
            #         weight = token_weight_pairs[i - 1][j][1]
            #         z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
        output.append(z)

    if (len(output) == 0):
        return z_empty, first_pooled
    return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()

def get_learned_conditioning(self, c):
    if self.cond_stage_forward is None:
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(c)
    else:
        assert hasattr(self.cond_stage_model, self.cond_stage_forward)
        c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
    return c

class FrozenCLIPEmbedderWithCustomWordsCustom(FrozenCLIPEmbedderWithCustomWords, SD1Tokenizer):
    '''
    Custom `FrozenCLIPEmbedderWithCustomWords` class that also inherits
    `SD1Tokenizer` to have the `_try_get_embedding()` method.

    Only supports SD1, not SD2 or SDXL.
    '''
    def populate_self_variables(self, from_):
        super_attrs = vars(from_)
        self_attrs = vars(self)
        self_attrs.update(super_attrs)

    def __init__(self, wrapped: SD1ClipModel, hijack):
        self.populate_self_variables(hijack.clip_orig.tokenizer)
        wrapped.tokenizer_backup = hijack.clip_orig.tokenizer # SD1Tokenizer

        # okay to modiy since CLIP was cloned
        wrapped.tokenizer = hijack.clip_orig.tokenizer.tokenizer # CLIPTokenizer.from_pretrained(tokenizer_path)
        # self.embedding_identifier_tokenized = hijack.clip_orig.tokenizer.tokenizer([self.embedding_identifier])["input_ids"][0][1:-1]
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens, return_pooled=False):
        return self.encode_from_tokens_comfy(tokens, return_pooled)

    def encode_with_transformers_comfy(self, tokens: List[List[int]], return_pooled=False) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        This function is different from `clip.cond_stage_model.encode_token_weights()`
        in that the tokens are `List[List[int]]`, not including the weights.

        Originally from `sd1_clip.py`: `encode()` -> `forward()`
        '''
        if type(tokens) == torch.Tensor:
            tokens = tokens.tolist()
        z, pooled = self.wrapped.encode(tokens)
        return (z, pooled) if return_pooled else z

    def encode_from_tokens_comfy(self, tokens: List[List[int]], return_pooled=False):
        '''
        The function is our rendition of `clip.encode_from_tokens()`.
        It still calls `clip.encode_from_tokens()` but hijacks the
        `clip.cond_stage_model.encode_token_weights()` method
        so we can run our own version of `encode_token_weights()`

        Originally from `sd.py`: `encode_from_tokens()`
        '''
        # note:
        # self.wrapped = self.hijack.clip_orig.cond_stage_model
        ret = None
        if type(tokens) == torch.Tensor:
            tokens = tokens.tolist()
        encode_token_weights_backup = self.hijack.clip_orig.cond_stage_model.encode_token_weights
        try:
            self.hijack.clip_orig.cond_stage_model.encode_token_weights = MethodType(self.encode_with_transformers_comfy, tokens)
            ret = self.hijack.clip_orig.encode_from_tokens(tokens, return_pooled)
            self.hijack.clip_orig.cond_stage_model.encode_token_weights = encode_token_weights_backup
        except Exception as error:
            self.hijack.clip_orig.cond_stage_model.encode_token_weights = encode_token_weights_backup
            raise error
        return ret


    def tokenize_line(self, line):

        self.parse_and_register_embeddings(line) # register embeddings, discard return

        return super().tokenize_line(line)

    # This function has been added to apply embeddings
    # from sd1_clip.py @ tokenize_with_weights()
    def parse_and_register_embeddings(self, text: str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        # parsed_weights = token_weights(text, 1.0)
        parsed = prompt_parser.parse_prompt_attention(text)
        parsed_weights = [tuple(tw) for tw in parsed]

        #tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            # to_tokenize = [x for x in weighted_segment if x != ""]

            tmp=[]
            # print(word)
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                emb_idx = word.find(self.embedding_identifier)
                # word.startswith(self.embedding_identifier)
                if emb_idx != -1 and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')

                    embedding_name_verbose = word[emb_idx:].strip('\n')
                    embedding_name = word[emb_idx+len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name.strip())
                    embed = embed.to(device=devices.device)

                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        self.hijack.embedding_db.register_embedding(Embedding(embed, embedding_name_verbose), self.hijack)
                        if len(embed.shape) == 1:
                            # tokens.append([(embed, weight)])
                            tmp += [(embed, weight)]
                        else:
                            # tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                            tmp += [(embed[x], weight) for x in range(embed.shape[0])]
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                # tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][1:-1]])
                tmp += [(word, weight)]
                # tokens.append(tmp)
                tokens += tmp
        return tokens


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum
