import regex as re
import random

from tqdm import tqdm

from MixTokenizer.core.utils import ChineseSplitter
from MixTokenizer.core.decode import HybridDecoder, ACAutomaton

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
        

        ZH_CHARS = {chr(i) for i in range(ZH_RANGE[0], ZH_RANGE[1] + 1)}
        ZH_USED_CHARS = set()

        instance.map_code = dict()
        instance.map_char = dict()
        instance.extra_length = 4

        temp_list = []
        temp_ch = []
        random.seed(42)
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
            temp_ch.append(ch)

        instance.use_ac = instance.use_ac or not all(len(x) == len(temp_list[0]) for x in temp_list)
        # instance.use_ac = False
        if not instance.use_ac:
            instance.mix_decoder = HybridDecoder(2 + instance.extra_length, temp_list)
        else:
            # use ac automaton for variable length patterns
            print("[INFO] Using AC Automaton for variable-length patterns in MixTokenizer.")
            instance.mix_decoder = ACAutomaton()
            for pattern, ch in zip(temp_list, temp_ch):
                instance.mix_decoder.add_pattern(pattern, ord(ch))
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
            ret = []
            for flag, l, r in self.splitter.py_split_zh_nonzh(token):
                if not flag:
                    t = "" .join(self.byte_encoder[b] for b in token[l:r].encode("utf-8"))
                else:
                    parts = []
                    for ch in token[l:r]:
                        enc = ch.encode("gb2312", errors="ignore")
                        parts.append(self.map_char[ch])
                        parts.extend(self.byte_encoder[b] for b in enc)
                    t = "".join(parts)
                ret.append(t)
                # print(f"segment: {token[l:r]}, is_zh: {flag}, {t}, len={len(t)}")
            # print(f"token after byte encoding: {ret}")
            ret = "".join(ret)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(ret).split(" "))
        return bpe_tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert a sequence of tokens into a decoded string (optimized)."""

        # join to one string (no need to split again)
        tokens_str = "".join(tokens)
        ords = [ord(ch) for ch in tokens_str]

        byte_decoder = self.byte_decoder  # local variable for speed
        errors = self.errors
        
        utf8 = "utf-8"
        gb = "gb2312"

        result_parts = []

        if not self.use_ac:
            intervals = self.mix_decoder.decode(ords, strict=False)
            extra_len = self.extra_length
            for flag, l, r in intervals:
                segment = tokens_str[l:r]
                if flag and extra_len:
                    # filter extra marker bytes
                    segment = "".join(
                        x for i, x in enumerate(segment) if i % (2 + extra_len) >= extra_len
                    )

                decoded_bytes = bytearray(byte_decoder[ch] for ch in segment)
                result_parts.append(decoded_bytes.decode(utf8 if not flag else gb, errors=errors))
        else:
            # use ac automaton to decode
            print(f"[INFO] Decoding using AC Automaton. {ords[:10]}...")
            intervals = self.mix_decoder.search(ords)
            print(f"[DEBUG] Found {len(intervals)} intervals during AC decoding.")
            for flag, l, r in intervals:
                if flag == -1:
                    segment = tokens_str[l:r]
                    decoded_bytes = bytearray(byte_decoder[ch] for ch in segment)
                    result_parts.append(decoded_bytes.decode(utf8, errors=errors))
                else:
                    result_parts.append(chr(flag))
                print(f"[DEBUG] Decoding segment from {l} to {r} with flag {flag}. Segment: {tokens_str[l:r]} flag={flag}, decoded='{result_parts[-1]}'")
        return "".join(result_parts)
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    
def get_mix_tokenizer(tokenizer_cls, dir_name="mix", use_ac=True):
    class_name = f"MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name, "use_ac": use_ac},
    )
    globals()[class_name] = mix_class
    return mix_class