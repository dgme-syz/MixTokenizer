import regex as re
import os
import json
from typing import List, Union, Optional, Dict
from pathlib import Path

import numpy as np
from bitarray import bitarray

from MixTokenizer.core.str import zh_encode

ZH_RANGE = (0x4E00, 0x9FFF)

EXTRA_ZH_CHARS = set([
    "。","？","！","【","】","，","、","；","：",
    "「","」","『","』","’","“","”","‘",
    "（","）","〔","〕","…","–","．","—",
    "《","》","〈","〉",
    "·","～","︰","︱",
])

def is_zh_char(c):
    cp = ord(c)
    if ZH_RANGE[0] <= cp <= ZH_RANGE[1]:
        return True
    return c in EXTRA_ZH_CHARS

def load_from_folder(path: Union[str, Path]) -> Dict[Path, bytes]:
    path = Path(path)
    files = {}
    for file_path in path.rglob("*"):  
        if file_path.is_file():
            files[file_path.relative_to(path)] = file_path.read_bytes()
    return files

def save_to_folder(path: Union[str, Path], files: Dict[Path, bytes]):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        file_path = path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)  
        file_path.write_bytes(content)


# PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
class Mystring:
    __slots__ = ("content", "flag")

    def __init__(self, content: str, flag: bool = False):
        if not isinstance(content, str):
            raise TypeError("content must be str")
        self.content = content
        self.flag = flag

    def __add__(self, other):
        if isinstance(other, Mystring):
            new_content = self.content + other.content
            new_flag = self.flag or other.flag
            return Mystring(new_content, new_flag)
        elif isinstance(other, str):
            return Mystring(self.content + str(other), self.flag)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return Mystring(str(other) + self.content, self.flag)
        else:
            return NotImplemented
    def __hash__(self):
        return hash(self.content)

    def __eq__(self, other):
        if isinstance(other, Mystring):
            return self.content == other.content
        elif isinstance(other, str):
            return self.content == str(other)
        return False

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class MixTokenizerBase:
    """
    Combines Qwen2Tokenizer with an additional tokenizer for a custom language.
    Allows mapping of new language tokens to composite representations.

    dummy class, need to register by a parent tokenizer
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # instance.pat = re.compile(PRETOKENIZE_REGEX)
        script_dir = os.path.join(pretrained_model_name_or_path, cls.dir_name)
        instance.save_cache = load_from_folder(script_dir)
        instance.vocab_len = instance.vocab_size // 2
        print(
            f"[INFO] vocab length = {instance.vocab_size}\n"
            f"[INFO] current length = {len(instance)}\n"
            f"[INFO] all_special_tokens = {instance.all_special_tokens}\n"
            f"[INFO] eos_token = {instance.eos_token}\n"
            f"[INFO] max_token_id, now = {instance.vocab_size - 1}\n"
        )
        return instance

    # dummy length
    def __len__(self):
        return 2 * super().__len__()
    
    @property
    def vocab_size(self):
        return len(self)

    def check_zh(self, token: str) -> bool:
        # print(f"token={token}")
        return is_zh_char(token)
    def check_zh_array(self, token_array: List[str]) -> List[bool]:
        return [is_zh_char(token) for token in token_array]

    def bpe(self, token_list: List):
        key = "".join([x.content for x in token_list])
        if key in self.cache:
            return self.cache[key]
         
        pairs = get_pairs(token_list)

        if not pairs:
            return " ".join([f"{x.content}" for x in token_list]), bitarray([x.flag for x in token_list])
        word = token_list
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # print(first.content, first.flag)
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i].content == first.content and i < len(word) - 1 and word[i + 1].content == second.content:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join([f"{x.content}" for x in word]), bitarray([x.flag for x in word])
        self.cache[key] = word
        return word

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token_list = []
            tokens, flags = zh_encode(token)
            tokens = [self.byte_encoder[b] for b in tokens]
            for t, f in zip(tokens, flags):
                token_list.append(Mystring(t, f))
            string, flag = self.bpe(token_list)
            bpe_tokens.extend([f"{s} {int(flg)}" for s, flg in zip(string.split(" "), flag)])
            del token_list, tokens, flags, string, flag
        # print(bpe_tokens)
        return bpe_tokens

    def _convert_one_token_to_id(self, token: str) -> Union[int, List[int]]:
        """
        Convert a token to one or more IDs depending on its type.
        """
        if " " in token and token != " ":
            token, flag = token.split()
            if flag == "1":
                # print(f"{token, flag}")
                temp = self._convert_token_to_id_with_added_voc(token)
                if isinstance(temp, int):
                    temp += self.vocab_len
                    return temp
                elif isinstance(temp, list):
                    return (np.array(temp) + self.vocab_len).tolist()
                else:
                    raise ValueError(f"expect type(token) = int or list!, get temp = {temp}")
            elif flag == "0":
                return self._convert_token_to_id_with_added_voc(token)
            else:
                raise ValueError(f"Invalid flag value: {flag}")
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
            if ids >= self.vocab_len:
                token_id = ids - self.vocab_len
            else:
                token_id = ids
            return super().convert_ids_to_tokens(token_id, skip_special_tokens=skip_special_tokens)
        ids = np.where(np.array(ids) >= self.vocab_len, np.array(ids) - self.vocab_len, np.array(ids)).tolist()
        return super().convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

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
        # patch, update tokenizer_config.json 
        tokenizer_config_path = Path(save_directory) / "tokenizer_config.json"

        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config.setdefault("auto_map", {})["AutoTokenizer"] = [
            f"{self.dir_name}/tokenizer.MixTokenizer",
            None
        ]
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
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