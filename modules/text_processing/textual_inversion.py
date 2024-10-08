import os
import torch
import base64
import json
import zlib
import logging
import numpy as np
import safetensors.torch
from PIL import Image


class EmbeddingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'TORCHTENSOR': obj.cpu().detach().numpy().tolist()}
        return json.JSONEncoder.default(self, obj)


class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, *args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, d):
        if 'TORCHTENSOR' in d:
            return torch.from_numpy(np.array(d['TORCHTENSOR']))
        return d


def embedding_to_b64(data):
    d = json.dumps(data, cls=EmbeddingEncoder)
    return base64.b64encode(d.encode())


def embedding_from_b64(data):
    d = base64.b64decode(data)
    return json.loads(d, cls=EmbeddingDecoder)


def lcg(m=2 ** 32, a=1664525, c=1013904223, seed=0):
    while True:
        seed = (a * seed + c) % m
        yield seed % 255


def xor_block(block):
    g = lcg()
    randblock = np.array([next(g) for _ in range(np.prod(block.shape))]).astype(np.uint8).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)


def crop_black(img, tol=0):
    mask = (img > tol).all(2)
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), mask.shape[1] - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), mask.shape[0] - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def extract_image_data_embed(image):
    d = 3
    outarr = crop_black(np.array(image.convert('RGB').getdata()).reshape(image.size[1], image.size[0], d).astype(np.uint8)) & 0x0F
    black_cols = np.where(np.sum(outarr, axis=(0, 2)) == 0)
    if black_cols[0].shape[0] < 2:
        print(f'{os.path.basename(getattr(image, "filename", "unknown image file"))}: no embedded information found.')
        return None

    data_block_lower = outarr[:, :black_cols[0].min(), :].astype(np.uint8)
    data_block_upper = outarr[:, black_cols[0].max() + 1:, :].astype(np.uint8)

    data_block_lower = xor_block(data_block_lower)
    data_block_upper = xor_block(data_block_upper)

    data_block = (data_block_upper << 4) | (data_block_lower)
    data_block = data_block.flatten().tobytes()

    data = zlib.decompress(data_block)
    return json.loads(data, cls=EmbeddingDecoder)


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False

        mt = os.path.getmtime(self.path)
        if self.mtime is None or mt > self.mtime:
            return True

    def update(self):
        if not os.path.isdir(self.path):
            return

        self.mtime = os.path.getmtime(self.path)


class EmbeddingDatabase:
    def __init__(self, tokenizer, expected_shape=-1):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.embedding_dirs = {}
        self.skipped_embeddings = {}
        self.expected_shape = expected_shape
        self.tokenizer = tokenizer
        self.fixes = []

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding):
        return self.register_embedding_by_name(embedding, embedding.name)

    def register_embedding_by_name(self, embedding, name):
        ids = self.tokenizer([name], truncation=False, add_special_tokens=False)["input_ids"][0]
        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        if name in self.word_embeddings:
            lookup = [x for x in self.ids_lookup[first_id] if x[1].name != name]
        else:
            lookup = self.ids_lookup[first_id]
        if embedding is not None:
            lookup += [(ids, embedding)]
        self.ids_lookup[first_id] = sorted(lookup, key=lambda x: len(x[0]), reverse=True)
        if embedding is None:
            if name in self.word_embeddings:
                del self.word_embeddings[name]
            if len(self.ids_lookup[first_id]) == 0:
                del self.ids_lookup[first_id]
            return None
        self.word_embeddings[name] = embedding
        return embedding

    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()

        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == '.PREVIEW':
                return

            embed_image = Image.open(path)
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                name = data.get('name', name)
            else:
                data = extract_image_data_embed(embed_image)
                if data:
                    name = data.get('name', name)
                else:
                    return
        elif ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            data = safetensors.torch.load_file(path, device="cpu")
        else:
            return

        emb_out = None
        if data is not None:
            embedding = create_embedding_from_data(data, name, filename=filename, filepath=path)

            # if self.expected_shape == -1 or self.expected_shape == embedding.shape:
            emb_out = self.register_embedding(embedding)
            # else:
            #    emb_out = self.skipped_embeddings[name] = embedding
        else:
            print(f"Unable to load Textual inversion embedding due to data issue: '{name}'.")
        return emb_out

    def load_from_dir(self, embdir):
        if not os.path.isdir(embdir.path):
            return

        for root, _, fns in os.walk(embdir.path, followlinks=True):
            for fn in fns:
                try:
                    fullfn = os.path.join(root, fn)

                    if os.stat(fullfn).st_size == 0:
                        continue

                    self.load_from_file(fullfn, fn)
                except Exception:
                    print(f"Error loading embedding {fn}")
                    continue

    def load_textual_inversion_embeddings(self):
        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()

        for embdir in self.embedding_dirs.values():
            self.load_from_dir(embdir)
            embdir.update()

        return

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding_from_data(data, name, filename='unknown embedding file', filepath=None):
    if 'string_to_param' in data:  # textual inversion embeddings
        param_dict = data['string_to_param']
        param_dict = getattr(param_dict, '_parameters', param_dict)  # fix for torch 1.12.1 loading saved file from torch 1.11
        assert len(param_dict) == 1, 'embedding file has multiple terms in it'
        emb = next(iter(param_dict.items()))[1]
        vec = emb.detach().to(dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    elif type(data) == dict and 'clip_g' in data and 'clip_l' in data:  # SDXL embedding
        vec = {k: v.detach().to(dtype=torch.float32) for k, v in data.items()}
        shape = data['clip_g'].shape[-1] + data['clip_l'].shape[-1]
        vectors = data['clip_g'].shape[0]
    elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:  # diffuser concepts
        assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

        emb = next(iter(data.values()))
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)
        vec = emb.detach().to(dtype=torch.float32)
        shape = vec.shape[-1]
        vectors = vec.shape[0]
    else:
        raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

    embedding = Embedding(vec, name)
    embedding.step = data.get('step', None)
    embedding.sd_checkpoint = data.get('sd_checkpoint', None)
    embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
    embedding.vectors = vectors
    embedding.shape = shape

    return embedding

from comfy.sd1_clip import expand_directory_list
def get_embed_file_path(embedding_name, embedding_directory):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except Exception:
            continue
        if not os.path.isfile(embed_path):
            extensions = ['.safetensors', '.pt', '.bin']
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    return valid_file

import re
from ..shared import logger
emb_re_ = r"(embedding:)?(?:({}[\w\.\-\!\$\/\\]+(\.safetensors|\.pt|\.bin)|(?(1)[\w\.\-\!\$\/\\]+|(?!)))(\.safetensors|\.pt|\.bin)?)(?:(:)(\d+\.?\d*|\d*\.\d+))?"
def get_valid_embeddings(embedding_directories):
    from  builtins import any as b_any
    exts = ['.safetensors', '.pt', '.bin']
    if isinstance(embedding_directories, str):
        embedding_directories = [embedding_directories]
    embedding_directories = expand_directory_list(embedding_directories)
    embs = set()
    from collections import OrderedDict, namedtuple
    EmbedInfo = namedtuple('EmbedInfo', ['basename', 'filename', 'filepath'])
    store = OrderedDict()
    for embd in embedding_directories:
        for root, dirs, files in os.walk(embd, followlinks=True, topdown=False):
            for name in files:
                if not b_any(x in os.path.splitext(name)[1] for x in exts): continue
                basename = os.path.basename(name)
                for ext in exts: basename=basename.removesuffix(ext)
                relpath_basename = os.path.normpath(os.path.join(os.path.relpath(root, embd), basename))
                k = os.path.normpath(os.path.join(os.path.relpath(root, embd), name))
                store[k] = EmbedInfo(basename, name, os.path.join(root, name))
                # add its counterpart
                if '/' in k:
                    store[k.replace('/', '\\')] = EmbedInfo(basename, name, os.path.join(root, name))
                elif '\\' in relpath_basename: 
                    store[k.replace('\\', '/')] = EmbedInfo(basename, name, os.path.join(root, name))
                
    embs = OrderedDict(sorted(store.items(), key=lambda item: len(item[0]), reverse=True))
    return embs

class EmbbeddingRegex:
    STR_PATTERN = r"(embedding:)?(?:({}[\w\.\-\!\$\/\\]+(\.safetensors|\.pt|\.bin)|(?(1)[\w\.\-\!\$\/\\]+|(?!)))(\.safetensors|\.pt|\.bin)?)(?:(:)(\d+\.?\d*|\d*\.\d+))?"
    def __init__(self, embedding_directory) -> None:
        self.embedding_directory = embedding_directory
        self.embeddings = get_valid_embeddings(self.embedding_directory)  if self.embedding_directory is not None else {}
        joined_keys = '|'.join([re.escape(os.path.splitext(k)[0]) for k in self.embeddings.keys()])
        emb_re = self.STR_PATTERN.format(joined_keys + '|' if joined_keys else '')
        self.pattern = re.compile(emb_re, flags=re.MULTILINE | re.UNICODE | re.IGNORECASE)

def parse_and_register_embeddings(self, text: str):
    embr = EmbbeddingRegex(self.embedding_directory)
    embs = embr.embeddings
    matches = embr.pattern.finditer(text)
    exts = ['.pt', '.safetensors', '.bin']

    for matchNum, match in enumerate(matches, start=1):
        found=False
        ext = (match.group(4) or (match.group(3) or ''))
        embedding_sname = (match.group(2) or '').removesuffix(ext)
        embedding_name = embedding_sname + ext
        if embedding_name:
            embed = None
            if ext:
                embed_info = embs.get(embedding_name + ext, None)
            else:
                for _ext in exts:
                    embed_info = embs.get(embedding_name + _ext, None)
                    if embed_info is not None: break
            if embed_info is not None:
                found=True
                try:
                    embed = self.embeddings.load_from_file(embed_info.filepath, embed_info.filename)
                except Exception as e:
                    logging.warning(f'\033[33mWarning\033[0m loading embedding `{embedding_name + ext}`: {e}')
            if embed is not None:
                found=True
                logger.debug(f'using embedding:{embedding_name}')
        if not found:
            logging.warning(f"\033[33mwarning\033[0m, embedding:{embedding_name} does not exist, ignoring")
    # ComfyUI trims non-existent embedding_names while A1111 doesn't.
    # here we get group 2,5,6. (group 2 minus its file extension)
    out = embr.pattern.sub(lambda m: (m.group(2) or '').removesuffix(m.group(4) or (m.group(3) or '')) + (m.group(5) or '') + (m.group(6) or ''), text)
    return out