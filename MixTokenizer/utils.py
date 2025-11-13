import regex as re
import random

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
        # for x, y in instance.byte_encoder.items():
        #     print(f"Byte encoder: {x} -> {ord(y)}")

        ZH_CHARS = {chr(i) for i in range(ZH_RANGE[0], ZH_RANGE[1] + 1)}
        ZH_USED_CHARS = set()

        instance.map_code = dict()
        instance.map_char = dict()
        instance.extra_length = 2
        for ch in tqdm(ZH_CHARS | EXTRA_ZH_CHARS, desc="Building HybridDecoder"):
            if isinstance(ch, int):
                ch = chr(ch)  
            try:
                bytes_seq = ch.encode("gb2312")
            except:
                # print(f"[WARN] cannot encode char {ch} in gb2312")
                continue
            
            if len(bytes_seq) != 2:
                assert False, f"Expected 2 bytes for gb2312 encoding of {ch}, got {len(bytes_seq)} bytes."

            if ch in ZH_USED_CHARS:
                continue
            ZH_USED_CHARS.add(ch if isinstance(ch, int) else ord(ch))
            instance.map_char[ch] = "".join([random.choice(list(instance.byte_encoder.values())) for _ in range(instance.extra_length)])
            instance.map_code[ch] = [ord(x) for x in instance.map_char[ch]]

            temp_list.append(instance.map_code[ch] + [ord(instance.byte_encoder[x]) for x in bytes_seq])
        instance.mix_decoder = HybridDecoder(2 + instance.extra_length, temp_list)
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
                t = ""
                for ch in token[l:r]:
                    add = "" if not flag else self.map_char[ch]
                    t += add + "".join(self.byte_encoder[b] for b in ch.encode("utf-8" if not flag else "gb2312"))
                ret += t
                # print(f"segment: {token[l:r]}, is_zh: {flag}, {t}, len={len(t)}")
            # print(f"token after byte encoding: {ret}")
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(ret).split(" "))
        return bpe_tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert a sequence of tokens into a decoded string (optimized)."""

        # join to one string (no need to split again)
        tokens_str = "".join(tokens)
        ords = [ord(ch) for ch in tokens_str]
        intervals = self.mix_decoder.decode(ords, strict=False)

        byte_decoder = self.byte_decoder  # local variable for speed
        errors = self.errors
        extra_len = self.extra_length
        utf8 = "utf-8"
        gb = "gb2312"

        result_parts = []

        for flag, l, r in intervals:
            segment = tokens_str[l:r]
            if flag and extra_len:
                # filter extra marker bytes
                segment = "".join(
                    x for i, x in enumerate(segment) if i % (2 + extra_len) >= extra_len
                )

            decoded_bytes = bytearray(byte_decoder[ch] for ch in segment)
            result_parts.append(decoded_bytes.decode(utf8 if not flag else gb, errors=errors))

        return "".join(result_parts)
    
    
def get_mix_tokenizer(tokenizer_cls, dir_name="mix"):
    class_name = f"MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name},
    )
    globals()[class_name] = mix_class
    return mix_class