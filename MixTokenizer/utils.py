import json
import os
from typing import List, Union, Dict, Optional
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer
from scipy.stats import qmc
from transformers.tokenization_utils import Trie

from MixTokenizer.core.utils import ChineseSplitter
from MixTokenizer.core.decode import HybridDecoder, ACAutomaton
from MixTokenizer.mapping import PrivateUnicodeMapper


def load_from_folder(path: str):
    path = Path(path)
    files = {}
    for file_path in path.iterdir():
        if file_path.is_file():
            files[file_path.name] = file_path.read_bytes()
    return files

def save_to_folder(path: str, files: Dict[str, bytes]):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for file_name, content in files.items():
        file_path = path / file_name
        file_path.write_bytes(content)

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
    
    def _convert_one_id_to_token(self, token_id: int) -> str:
        x = self.tokenizer.id_to_token(token_id)
        return x

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
        # Load extra_config.json
        script_dir = os.path.join(pretrained_model_name_or_path, cls.dir_name)
        instance.save_cache = load_from_folder(script_dir)
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
        instance.splitter = ChineseSplitter(list(instance.MAPPER.mapping.values()))
        instance.map_back=True

        instance.new_lang_tokenizer = new_lang_tokenizer
        level = extra_config.get("level", None)

        if extra_config.get("mapping") and extra_config.get("used_ids"):
            print(f"Mapping file and used ids are loaded, level ignored: {level}")
            instance.mapping_list = extra_config["mapping"]

            equal_length = all(len(x) == len(instance.mapping_list[0]) for x in instance.mapping_list)
            if equal_length:
                print("[INFO] Using Hash Decoder for fixed-length patterns in MixTokenizer.")
                instance.mix_decoder = HybridDecoder(len(instance.mapping_list[0]))
            else:
                print("[INFO] Using AC Automaton for variable-length patterns in MixTokenizer.")
                instance.mix_decoder = ACAutomaton()
            
            # add pattern
            for i, pattern in enumerate(instance.mapping_list):
                instance.mix_decoder.add_pattern(pattern, i)
            if not equal_length:
                instance.mix_decoder.build()

            instance.level = len(instance.mapping_list[0])
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
        else:
            raise ValueError(
                "Ensure mapping and used_ids exist in config, or frequency_id_files and level exist in config."
            )
        
        print(
            f"[INFO] current length = {instance.vocab_size}\n"
            f"[INFO] all_special_tokens = {instance.all_special_tokens}\n"
            f"[INFO] eos_token = {instance.eos_token}\n"
        )
        return instance
    
    def __len__(self):
        return max(super().__len__(), (max(self.zero_ids) + 1) if hasattr(self, "zero_ids") else 0)
    
    @property
    def vocab_size(self):
        return len(self) # support vllm
    
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Two-stage tokenization:
        1. Group characters by type (new_lang vs Qwen).
        2. Tokenize each segment using the appropriate tokenizer.
        """
        # First chunk
        # print(f"tokenize text={text}")
        # In this setting, src can be normal Chinese text.
        text = self.MAPPER.map_string(text)
        tokens = []
        append = tokens.extend  
        for chunk_text in self.new_lang_tokenizer.split_by_special_tokens(text):
            if chunk_text in self.new_lang_tokenizer.all_special_tokens:
                append(self.new_lang_tokenizer.tokenize(chunk_text))
            else:
                for flag, start, end in self.splitter.py_split_zh_nonzh(chunk_text):
                    seg = chunk_text[start:end]
                    if len(seg) == 0: 
                        continue
                    if flag:
                        append(self.new_lang_tokenizer.tokenize(seg))
                    else:
                        append(super().tokenize(seg, **kwargs))
        return tokens

    def _convert_one_token_to_id(self, token: str) -> Union[int, List[int]]:
        """
        Convert a token to one or more IDs depending on its type.
        """
        if self.splitter.py_split_zh_nonzh(token)[0][0]:
            token_id = self.new_lang_tokenizer._convert_one_token_to_id(token)
            return self.mapping_list[token_id]
        else:
            return self._convert_token_to_id_with_added_voc(token)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_one_token_to_id(tokens)

        ids = []
        for token in tokens:
            mapped = self._convert_one_token_to_id(token)
            ids.extend(mapped if isinstance(mapped, list) else [mapped])
        return ids

    def convert_ids_to_tokens(
        self, ids: Union[int, list[int]], skip_special_tokens: bool = False
    ) -> Union[str, list[str]]:
        if isinstance(ids, int):
            ids = [ids]
        if not ids:
            return ""
        
        decoded_segments = []
        append = decoded_segments.append
        extend = decoded_segments.extend
        for flag, start, end in self.mix_decoder.search(ids):
            seg_ids = ids[start:end]
            if flag == -1:
                extend(super().convert_ids_to_tokens(seg_ids, skip_special_tokens=skip_special_tokens))
            else:
                ch = self.new_lang_tokenizer._convert_one_id_to_token(flag)
                append(ch)

        return decoded_segments
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens back to a string.
        """
        map_back = getattr(self, "map_back", True)
        text = ""
        raw_tokens = []
        for token in tokens:
            if self.splitter.py_split_zh_nonzh(token)[0][0]:
                if raw_tokens:
                    segment = super().convert_tokens_to_string(raw_tokens)
                    text += segment
                    raw_tokens = []
                if map_back:
                    token = self.MAPPER.unmap_string(token)
                text += token
            else:
                raw_tokens.append(token)
        if raw_tokens:
            segment = super().convert_tokens_to_string(raw_tokens)
            text += segment
        return text
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> tuple[str, ...]:
        save_files = super().save_pretrained(
            save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        # save mix inject dir
        mix_dir = os.path.join(save_directory, self.dir_name)
        save_to_folder(mix_dir, self.save_cache)
        return save_files

    
def get_mix_tokenizer(tokenizer_cls, dir_name="mix"):
    class_name = f"MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name},
    )
    globals()[class_name] = mix_class
    return mix_class