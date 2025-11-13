import regex as re
import random, hashlib
from typing import List
import os

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

def generate_non_substring_sequences_double_hash(
    alphabet,
    k=30000,
    min_len=2,
    max_len=5,
    seed=0
):
    rng = random.Random(seed)
    ids = [ord(x) for x in alphabet]
    max_id = max(ids)
    results = []
    used_hashes = set() 

    def rolling_hash_double(seq):
        hashes = []
        base1 = max_id + 1
        mod1 = 10**18 + 7
        base2 = max_id + 3
        mod2 = 10**18 + 9
        n = len(seq)
        for l in range(2, n+1): 
            h1 = h2 = 0
  
            for i in range(l):
                h1 = (h1 * base1 + seq[i]) % mod1
                h2 = (h2 * base2 + seq[i]) % mod2
            hashes.append((h1, h2))

            pow1 = pow(base1, l-1, mod1)
            pow2 = pow(base2, l-1, mod2)
            for i in range(1, n-l+1):
                h1 = (h1 - seq[i-1]*pow1) % mod1
                h1 = (h1 * base1 + seq[i+l-1]) % mod1
                h2 = (h2 - seq[i-1]*pow2) % mod2
                h2 = (h2 * base2 + seq[i+l-1]) % mod2
                hashes.append((h1, h2))
        return hashes

    pbar = tqdm(total=k, desc="Generating sequences")
    while len(results) < k:
        L = rng.randint(min_len, max_len)
        seq = [rng.choice(ids) for _ in range(L)]
        hashes = rolling_hash_double(seq)
        if any(h in used_hashes for h in hashes):
            continue 
        results.append(seq)
        for h in hashes:
            used_hashes.add(h)
        pbar.update(1)
    pbar.close()
    return results


class MixTokenizerBase:
    """
    Combines Qwen2Tokenizer with an additional tokenizer for a custom language.
    Allows mapping of new language tokens to composite representations.

    dummy class, need to register by a parent tokenizer
    """
    use_code: str = None  # encoding to use, e.g., "gb2312"
    use_ac: bool = False  # whether to use AC automaton for variable-length patterns
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        

        ZH_CHARS = {chr(i) for i in range(ZH_RANGE[0], ZH_RANGE[1] + 1)}
        ZH_USED_CHARS = set()


        temp_list = []
        temp_ch = []
        for ch in tqdm(ZH_CHARS | EXTRA_ZH_CHARS, desc="Collecting Chinese characters"):
            if isinstance(ch, int):
                ch = chr(ch)  
            if instance.use_code:
                try:
                    ch.encode(instance.use_code) # test encoding
                except:
                    continue

            if ch in ZH_USED_CHARS:
                continue
            ZH_USED_CHARS.add(ch if isinstance(ch, int) else ord(ch))
        ZH_USED_CHARS = list(ZH_USED_CHARS)
        # random or some deterministic mapping
        instance.use_map = dict()
        if instance.use_code is None:
            # random mapping
            print("[INFO] Generating random code mappings for Chinese characters.")
            alpahabet = "".join(instance.byte_encoder.values())
            script_dir = os.path.join(pretrained_model_name_or_path, instance.dir_name)
            mapped_file = os.path.join(script_dir, "encode.txt")
            _flag = False
            if os.path.exists(mapped_file):
                print(f"[INFO] Loading existing code mappings from {mapped_file}.")
                with open(mapped_file, "r", encoding="utf-8") as f:
                    for line in f:
                        code, ch = line.strip().split("\t")
                        temp_list.append([ord(b) for b in code])
                        temp_ch.append(ord(ch))
                max_len = max([len(x) for x in temp_list])
                min_len = min([len(x) for x in temp_list])
                if len(temp_ch) == len(list(ZH_USED_CHARS)) and max_len == getattr(instance, "max_code_len", 5) and min_len == getattr(instance, "min_code_len", 2):
                    _flag = True
                else:
                    _flag = False
                print(f"[INFO] Loaded {len(temp_ch)} mappings. Code length range: {min_len} to {max_len}. flag={_flag}")
                    
            if not _flag:
                print(f"[INFO] Generating new code mappings and saving to {mapped_file}.")
                print(f"[INFO] Total Chinese characters to map: {len(ZH_USED_CHARS)}")
                print(f"[INFO] Generating codes with length from {getattr(instance, 'min_code_len', 2)} to {getattr(instance, 'max_code_len', 5)}")
                temp_list = generate_non_substring_sequences_double_hash(
                    alpahabet,
                    k=len(ZH_USED_CHARS),
                    min_len=getattr(instance, "min_code_len", 2),
                    max_len=getattr(instance, "max_code_len", 5),
                    seed=42,
                )
                # save to file
                temp_ch = list(ZH_USED_CHARS)
                os.makedirs(script_dir, exist_ok=True)
                with open(mapped_file, "w", encoding="utf-8") as f:
                    for code, ch in zip(temp_list, temp_ch):
                        f.write(f'{"".join(chr(b) for b in code)}\t{chr(ch)}\n')
                print(f"[INFO] Saved generated code mappings to {mapped_file}.")
                
        else:
            assert isinstance(instance.use_code, str)
            print(f"[INFO] Using predefined code file: {instance.use_code}")
            temp_ch = []
            temp_rd = []
            if getattr(instance, "use_code_random", True):
                script_dir = os.path.join(pretrained_model_name_or_path, instance.dir_name)
                rd_file = os.path.join(script_dir, f"code_random_{instance.use_code}.txt")
                _flag = False
                if os.path.exists(rd_file):
                    print(f"[INFO] Loading existing random code mappings from {rd_file}.")
                    with open(rd_file, "r", encoding="utf-8") as f:
                        for line in f:
                            temp_rd.append([ord(b) for b in line.strip()])
                    if len(temp_rd) == len(list(ZH_USED_CHARS)) and all(len(x) == getattr(instance, "code_random_len", 2) for x in temp_rd):
                        _flag = True
                    print(f"[INFO] Loaded {len(temp_rd)} random mappings. flag={_flag} extra_len={len(temp_rd[0]) if temp_rd else 0}")
                if not _flag:
                    print(f"[INFO] Generating new random code mappings and saving to {rd_file}.")
                    print(f"[INFO] Total Chinese characters to map: {len(ZH_USED_CHARS)}")
                    print(f"[INFO] Generating random codes with length {getattr(instance, 'code_random_len', 2)}")
                    alpahabet = "".join(instance.byte_encoder.values())
                    temp_rd = [random.choices(alpahabet, k=getattr(instance, "code_random_len", 2)) for _ in range(len(ZH_USED_CHARS))]
                    with open(rd_file, "w", encoding="utf-8") as f:
                        for code in temp_rd:
                            f.write(f'{"".join(code)}\n')
                    temp_rd = [[ord(b) for b in code] for code in temp_rd]
                    
            for rd, ch in zip(temp_rd, ZH_USED_CHARS):
                code = chr(ch).encode(instance.use_code)
                temp_list.append(rd + [ord(instance.byte_encoder[b]) for b in code])
                temp_ch.append(ch)
        instance.use_ac = instance.use_ac or not all(len(x) == len(temp_list[0]) for x in temp_list)
        # instance.use_ac = False
        if not instance.use_ac:
            print("[INFO] Using Hash Decoder for fixed-length patterns in MixTokenizer.")
            instance.mix_decoder = HybridDecoder(len(temp_list[0]))
            # print(temp_list[:2])
            for pattern, ch in zip(temp_list, temp_ch):
                # print("[DEBUG] pattern:", pattern, "ch:", ch)
                assert all(chr(x) in instance.byte_encoder.values() for x in pattern)
                instance.use_map[chr(ch)] = "".join(chr(b) for b in pattern)
                instance.mix_decoder.add_pattern(pattern, ch)
        else:
            # use ac automaton for variable length patterns
            print("[INFO] Using AC Automaton for variable-length patterns in MixTokenizer.")
            instance.mix_decoder = ACAutomaton()
            for pattern, ch in zip(temp_list, temp_ch):
                assert all(chr(x) in instance.byte_encoder.values() for x in pattern)
                instance.use_map[chr(ch)] = "".join(chr(b) for b in pattern)
                instance.mix_decoder.add_pattern(pattern, ch)
            instance.mix_decoder.build()
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
                        parts.append(self.use_map[ch])
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

        intervals = self.mix_decoder.search(ords)
        for flag, l, r in intervals:
            if flag == -1:
                segment = tokens_str[l:r]
                decoded_bytes = bytearray(byte_decoder[ch] for ch in segment)
                result_parts.append(decoded_bytes.decode(utf8, errors=errors))
            else:
                result_parts.append(chr(flag))
            # print(f"[DEBUG] Decoding segment from {l} to {r} with flag {flag}. Segment: {tokens_str[l:r]} flag={flag}, decoded='{result_parts[-1]}'")
        return "".join(result_parts)
  
    
def get_mix_tokenizer(tokenizer_cls, dir_name="mix", use_ac=False, use_code="gbk", **kwargs):
    kwargs["min_code_len"] = 4
    kwargs["max_code_len"] = 4
    kwargs["use_code_random"] = True
    kwargs["code_random_len"] = 2
    class_name = f"MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name, "use_ac": use_ac, "use_code": use_code, **kwargs},
    )
    globals()[class_name] = mix_class
    return mix_class


# decode module
## ac automaton: support variable-length patterns
## hash decoder: support fixed-length patterns only

# encode module
## fully random
# --min_code_len
# --max_code_len

## code-based (gb2312, gbk, big5, etc.)
# --use_code
# --use_code_random
# --code_random_len (add extra random bytes)

## [Tip]
### fully random: length >= 4 is recommended
### code-based: code_random_len >= 2 is recommended