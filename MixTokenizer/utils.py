import json
import os
from typing import List, Union, Dict

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer
from scipy.stats import qmc
from transformers.tokenization_utils import Trie

from MixTokenizer.core.cpp_core import find_change_points, is_new_char_array, ComboTrie
from MixTokenizer.lang_map.mapping import PrivateUnicodeMapper

def sample_integer_points(L: int, K: int, N: int, seed: int = 42) -> np.ndarray:
    """
    Sample N integer points within [0, L-1]^K using a quasi-random Sobol sequence.

    Args:
        L (int): Range of each dimension [0, L-1].
        K (int): Number of dimensions.
        N (int): Number of points to sample.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: (N, K) integer array of sampled points.
    """
    if seed is not None:
        np.random.seed(seed)

    # Sobol sequence produces low-discrepancy, quasi-uniform samples
    sampler = qmc.Sobol(d=K, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(N)))  # Sobol requires 2^m samples
    sobol_points = sampler.random_base2(m=m)[:N]

    int_points = np.floor(sobol_points * L).astype(int)
    np.random.shuffle(int_points)
    return int_points

class NewLangTokenizer:
    """
    Wrapper around BertWordPieceTokenizer for a custom language or symbol set.
    """

    def __init__(self, vocab_file: str | dict, **kwargs) -> None:
        if isinstance(vocab_file, str):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
        else:
            vocab = vocab_file
        self.unk_token = "[UNK]"
        self.tokens_trie = Trie()
        self.all_special_tokens = []
        for mark in [self.unk_token]:
            self.tokens_trie.add(mark)
            self.all_special_tokens.append(mark)
            if mark not in vocab:
                vocab[mark] = len(vocab)
                
        self.tokenizer = Tokenizer(WordLevel(vocab, unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated") #important

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def tokenize(self, text: str) -> List[str]:
        x = self.tokenizer.encode(text, add_special_tokens=False).tokens
        return x

    def _convert_one_token_to_id(self, token: str) -> Union[int, List[int]]:
        x = self.tokenizer.token_to_id(token)
        return x 
    
    def is_new_char(self, ch: str) -> bool:
        # Only treat single-character entries in vocab as new characters
        # For faster tokenize, just ensure in private area

        status = (
            ch in self.all_special_tokens or (
                len(ch) == 1 and ( 
                    0xE000 <= ord(ch) <= 0xF8FF
                    or 0xF0000 <= ord(ch) <= 0xFFFFD
                    or 0x100000 <= ord(ch) <= 0x10FFFD
                )
            ) 
        )

        return status
    
    def is_new_char_array(self, text: str) -> np.ndarray:
        if text in self.all_special_tokens:
            return np.ones(len(text), dtype=bool)
        return is_new_char_array(text)

    def split_by_special_tokens(self, text: str) -> List[str]:
        return self.tokens_trie.split(text)

    def decode(self, token_ids: List[int]) -> str:
        # Decode and remove spaces introduced by WordPiece
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
    
    @property
    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def save_pretrained(self, save_directory: str) -> None:
        vocab = self.tokenizer.get_vocab()
        # sort by value
        vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    


class MixTokenizerBase:
    """
    Combines Qwen2Tokenizer with an additional tokenizer for a custom language.
    Allows mapping of new language tokens to composite representations.

    dummy class, need to register by a parent tokenizer
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        instance.pretrained_model_name_or_path = pretrained_model_name_or_path
        # Load extra_config.json
        script_dir = os.path.join(pretrained_model_name_or_path, cls.dir_name)
        json_path = os.path.join(script_dir, "extra_config.json")
        with open(json_path, "r", encoding="utf-8") as f:
            extra_config = json.load(f)

        # Load new tokenizer
        new_path = os.path.join(script_dir, "new_tokenizer")
        try:
            print("Try to load AutoTokenizer from HF")
            new_lang_tokenizer = AutoTokenizer.from_pretrained(new_path)
        except Exception:
            print("Fallback: use default WordLevel Tokenizer")
            vocab_path = os.path.join(new_path, "vocab.json")
            new_lang_tokenizer = NewLangTokenizer(vocab_file=vocab_path)

        # Load old2new mapping
        map_path = os.path.join(script_dir, "lang_map", "mapping.json")
        instance.MAPPER = PrivateUnicodeMapper.from_mapping_dict(map_path)

        instance.new_lang_tokenizer = new_lang_tokenizer
        level = extra_config.get("level", None)

        if extra_config.get("mapping") and extra_config.get("used_ids"):
            print(f"Mapping file and used ids are loaded, level ignored: {level}")
            instance.mapping = extra_config["mapping"]
            instance.trie = ComboTrie()
            for x in instance.mapping:
                instance.trie.insert(x)
            instance.level = len(instance.mapping[0])
            instance.zero_ids = extra_config["used_ids"]
            # ensure zero_ids are not in special_ids
            if not hasattr(instance, "all_special_ids"):
                instance.all_special_ids = [
                    instance.convert_tokens_to_ids(t) for t in instance.all_special_tokens
                ]
                print(f"We collect all_special_ids = {instance.all_special_ids}")
            assert not set(instance.zero_ids).intersection(set(instance.all_special_ids)), \
                "zero_ids should not overlap with special token ids."
            vocab_len = len(instance) + 10 # for safety
            instance.zero_mask = np.zeros(vocab_len, dtype=bool)
            instance.zero_mask[instance.zero_ids] = True
            instance.reverse_mapping = {}
            for i, point in enumerate(instance.mapping):
                instance.reverse_mapping[tuple(point)] = i
            # ensure mapping and reverse_mapping are consistent
            for k, v in instance.reverse_mapping.items():
                assert tuple(instance.mapping[v]) == k, "Mapping and reverse mapping are inconsistent."
        else:
            raise ValueError(
                "Ensure mapping and used_ids exist in config, or frequency_id_files and level exist in config."
            )
        
        print(
            f"[INFO] current length = {len(instance)}\n"
            f"[INFO] all_special_tokens = {instance.all_special_tokens}\n"
            f"[INFO] eos_token = {instance.eos_token}\n"
        )
        return instance
    
    def __len__(self):
        return max(super().__len__(), (max(self.zero_ids) + 1) if hasattr(self, "zero_ids") else 0)

    def _convert_ids_to_new_lang_id(self, token_ids: List[int] | np.ndarray) -> int:
        return self.reverse_mapping[tuple(token_ids)]

    def _convert_ids_to_new_lang_id_batch(self, batch_token_ids: List[List[int]] | np.ndarray) -> List[int]:
        for tids in batch_token_ids:
            if list(tids) not in self.mapping:
                raise ValueError("acnachjakchas")
        return [self.reverse_mapping[tuple(tids)] for tids in batch_token_ids]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Two-stage tokenization:
        1. Group characters by type (new_lang vs Qwen).
        2. Tokenize each segment using the appropriate tokenizer.
        """
        # First chunk
        # print(f"tokenize text={text}")
        tokens = []
        append = tokens.extend  
        for chunk_text in self.new_lang_tokenizer.split_by_special_tokens(text):
            if chunk_text in self.new_lang_tokenizer.all_special_tokens:
                append(self.new_lang_tokenizer.tokenize(chunk_text))
            else:
                is_new = self.new_lang_tokenizer.is_new_char_array(chunk_text)

                # transfer [0,0,1,1,0] → [0,2,4,5]
                # True: new_lang seg, False: base_lang seg
                segments = find_change_points(is_new)

                for flag, start, end in segments:
                    seg = chunk_text[start:end]
                    if len(seg) == 0: 
                        continue
                    if flag:
                        append(self.new_lang_tokenizer.tokenize(seg))
                    else:
                        append(super().tokenize(seg, **kwargs))
        # print(f"tokenized tokens={tokens}")
        return tokens

    def _convert_one_token_to_id(self, token: str) -> Union[int, List[int]]:
        """
        Convert a token to one or more IDs depending on its type.
        """
        if self.new_lang_tokenizer.is_new_char(token):
            token_id = self.new_lang_tokenizer._convert_one_token_to_id(token)
            # print("***||")
            # print(token_id, self.mapping[token_id])
            return self.mapping[token_id]
        else:
            return self._convert_token_to_id_with_added_voc(token)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        # print(f"tokens = {tokens}")
        if isinstance(tokens, str):
            return self._convert_one_token_to_id(tokens)

        ids = []
        for token in tokens:
            mapped = self._convert_one_token_to_id(token)
            ids.extend(mapped if isinstance(mapped, list) else [mapped])
        return ids

    def _decode(self, token_ids: Union[int, List[int]], **kwargs) -> str:
        """
        High-performance decoding of token ID sequences.
        Groups consecutive tokens by type (new_lang vs base_lang),
        then decodes each block using the appropriate tokenizer.
        """

        map_back = kwargs.get("map_back", True)

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if not token_ids:
            return ""
        
        segments = self.trie.find_change_points_plus(token_ids)
        token_ids = np.array(token_ids, dtype=np.int64)
        # print("dxcadasdas")
        decoded_segments = []
        append = decoded_segments.append
        decode_new = self.new_lang_tokenizer.decode
        rev_map = self._convert_ids_to_new_lang_id_batch
        lvl = self.level
        # print(segments)
        for flag, start, end in segments:
            seg_ids = token_ids[start:end]
            if len(seg_ids) == 0:
                continue
            if flag:
                # print(seg_ids)
                assert len(seg_ids) % lvl == 0, f"Invalid new language token length {len(seg_ids)} (level={lvl})"
                # chunk level
                grouped = rev_map(seg_ids.reshape(-1, lvl))
                ans = decode_new(grouped)
                if map_back:
                    # print("mapppp--------------------------------")
                    # print(ans)
                    ans = self.MAPPER.unmap_string(ans)
                    # print(ans)
                append(ans)
            else:
                # print("dapppp--------------------------------")
                # print(seg_ids)
                append(super()._decode(seg_ids.tolist(), **kwargs))

        return "".join(decoded_segments)
    
def get_mix_tokenizer(tokenizer_cls, dir_name="mix"):
    class_name = f"MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name},
    )
    globals()[class_name] = mix_class
    return mix_class