import regex as re
from typing import List, Union

import numpy as np
from tqdm import tqdm

from MixTokenizer.core.utils import ChineseSplitter
from MixTokenizer.core.decode import HybridDecoder

ZH_RANGE = (0x4E00, 0x9FFF)

EXTRA_ZH_CHARS = set([
    "。","？","！","【","】","，","、","；","：",
    "「","」","『","』","’","“","”","‘",
    "（","）","〔","〕","…","–","．","—",
    "《","》","〈","〉",
    "·","～","︰","︱",
])

class MixTokenizerBase:
    """
    Combines Qwen2Tokenizer with an additional tokenizer for a custom language.
    Allows mapping of new language tokens to composite representations.

    dummy class, need to register by a parent tokenizer
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        temp_list = []
        ZH_CHARS = {chr(i) for i in range(ZH_RANGE[0], ZH_RANGE[1] + 1)}
        ZH_USED_CHARS = set()
        for ch in tqdm(ZH_CHARS | EXTRA_ZH_CHARS, desc="Building HybridDecoder"):
            if isinstance(ch, int):
                ch = chr(ch)  
            try:
                bytes_seq = ch.encode("gb2312")
            except:
                print(f"[WARN] cannot encode char {ch} in gb2312")
                continue
            
            if len(bytes_seq) != 2:
                assert False, f"Expected 2 bytes for gb2312 encoding of {ch}, got {len(bytes_seq)} bytes."
            ZH_USED_CHARS.add(ch if isinstance(ch, int) else ord(ch))
            temp_list.append([ord(instance.byte_encoder[x]) for x in bytes_seq])
        instance.mix_decoder = HybridDecoder(2, temp_list)
        instance.splitter = ChineseSplitter(list(ZH_USED_CHARS))
        print(
            f"[INFO] current length = {len(instance)}\n"
            f"[INFO] all_special_tokens = {instance.all_special_tokens}\n"
            f"[INFO] eos_token = {instance.eos_token}\n"
        )
        return instance

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            ret = ""
            for flag, l, r in self.splitter.py_split_zh_nonzh(token):
                t = "".join(
                    self.byte_encoder[b] for b in token[l:r].encode("utf-8" if not flag else "gb2312")
                )
                ret += t
                print(f"segment: {token[l:r]}, is_zh: {flag}, {t}, len={len(t)}")
            # print(f"token after byte encoding: {ret}")
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(ret).split(" "))
        return bpe_tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        tokens = [x  for x in "".join(tokens)]
        intervals = self.mix_decoder.decode([ord(x) for x in tokens], strict=False)
        ans = ""
        for flag, l, r in intervals:
            temp = bytearray(
                [self.byte_decoder[c] for c in "".join(tokens[l:r])]
            ).decode("utf-8" if not flag else "gb2312", errors=self.errors)
            ans += temp
            print(f"Decode segment: {tokens[l:r]}, is_zh: {flag}, l={l}, r={r} -> {temp}")

        return ans
    
    
def get_mix_tokenizer(tokenizer_cls, dir_name="mix"):
    class_name = f"MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name},
    )
    globals()[class_name] = mix_class
    return mix_class