import regex as re
from typing import List, Union

import numpy as np

ZH_RE = re.compile(
    r'[\u4e00-\u9fff'
    r'\u3002\uFF1F\uFF01\u3010\u3011\uFF0C\u3001\uFF1B\uFF1A'
    r'\u300C\u300D\u300E\u300F\u2019\u201C\u201D\u2018'
    r'\uFF08\uFF09\u3014\u3015\u2026\u2013\uFF0E\u2014'
    r'\u300A\u300B\u3008\u3009'
    r'\u00B7\uFF5E\uFE30\uFE31]'
) # not contain space

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
        instance.vocab_len = super().__len__(instance)
        # instance.pat = re.compile(PRETOKENIZE_REGEX)
        print(
            f"[INFO] vocab length = {instance.vocab_len}\n"
            f"[INFO] current length = {len(instance)}\n"
            f"[INFO] all_special_tokens = {instance.all_special_tokens}\n"
            f"[INFO] eos_token = {instance.eos_token}\n"
        )
        return instance

    # dummy length
    def __len__(self):
        return 2 * super().__len__()

    def check_zh(self, token: str) -> bool:
        # print(f"token={token}")
        return ZH_RE.search(token) and token not in self.all_special_tokens

    def bpe(self, token_list: List):
        key = tuple(token_list)
        if key in self.cache:
            return self.cache[key]
         
        pairs = get_pairs(token_list)

        if not pairs:
            return [f"{x.content} {x.flag}" for x in token_list]
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
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
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
        word = [f"{x.content} {x.flag}" for x in word]
        self.cache[key] = word
        return word

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token_list = []
            for ch in token:
                if self.check_zh(ch):
                    token_list.extend([ Mystring(self.byte_encoder[b], 1) for b in ch.encode("utf-8") ])
                else:
                    token_list.extend([ Mystring(self.byte_encoder[b], 0) for b in ch.encode("utf-8") ])

            bpe_tokens.extend(self.bpe(token_list))
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
                raise ValueError
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

    def _decode(self, token_ids: Union[int, List[int]], **kwargs) -> str:
        """
        High-performance decoding of token ID sequences.
        Groups consecutive tokens by type (new_lang vs base_lang),
        then decodes each block using the appropriate tokenizer.
        """
        arr = np.array(token_ids)
        token_ids = np.where(arr >= self.vocab_len, arr - self.vocab_len, arr).tolist()
        return super()._decode(token_ids, **kwargs)
    
def get_mix_tokenizer(tokenizer_cls, dir_name="mix"):
    class_name = f"MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name},
    )
    globals()[class_name] = mix_class
    return mix_class