import regex as re
import random
from typing import List, Optional, Callable, Union, Dict
from pathlib import Path
import yaml
import os
import json

from tqdm import tqdm

from MixTokenizer.core.utils import ChineseSplitter
from MixTokenizer.core.decode import HybridDecoder, ACAutomaton

ZH_RANGE = (0x4E00, 0x9FFF)

EXTRA_ZH_CHARS = {
    "。","？","！","【","】","，","、","；","：",
    "「","」","『","』","’","“","”","‘",
    "（","）","〔","〕","…","–","．","—",
    "《","》","〈","〉",
    "·","～","︰","︱",
}


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

def generate_non_substring_sequences_double_hash(
    alphabet: str,
    k: int = 30000,
    min_len: int = 2,
    max_len: int = 5,
    seed: int = 0,
) -> List[List[int]]:
    """Generate k sequences from the given alphabet such that no sequence is a substring of another.

    The function uses a pair of rolling hashes to detect substring collisions.
    Behavior and return value are unchanged from the original implementation.
    """
    rng = random.Random(seed)
    ids = [ord(x) for x in alphabet]
    max_id = max(ids)
    results: List[List[int]] = []
    used_hashes = set()

    def rolling_hash_double(seq: List[int]):
        hashes = []
        base1 = max_id + 1
        mod1 = 10**18 + 7
        base2 = max_id + 3
        mod2 = 10**18 + 9
        n = len(seq)
        for l in range(2, n + 1):
            h1 = h2 = 0

            for i in range(l):
                h1 = (h1 * base1 + seq[i]) % mod1
                h2 = (h2 * base2 + seq[i]) % mod2
            hashes.append((h1, h2))

            pow1 = pow(base1, l - 1, mod1)
            pow2 = pow(base2, l - 1, mod2)
            for i in range(1, n - l + 1):
                h1 = (h1 - seq[i - 1] * pow1) % mod1
                h1 = (h1 * base1 + seq[i + l - 1]) % mod1
                h2 = (h2 - seq[i - 1] * pow2) % mod2
                h2 = (h2 * base2 + seq[i + l - 1]) % mod2
                hashes.append((h1, h2))
        return hashes

    with tqdm(total=k, desc="Generating sequences") as pbar:
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
    return results


def ask(prompt: str, default=None, cast: Optional[Callable] = None):
    """Modern input helper.

    - prompt: text shown to user
    - default: default value returned when the user presses Enter
    - cast: optional callable to cast the raw input; returns default on failure
    """
    raw = input(f"{prompt} ").strip()
    if not raw:
        return default
    if cast:
        try:
            return cast(raw)
        except Exception:
            print(f"Invalid input, using default: {default}")
            return default
    return raw


def create_config(path: str) -> None:
    print("=== MixTokenizer Config Creator ===")

    config = {}

    use_code = ask("Encoding to use (e.g. gbk; Or 'random' for random encoding):", default=None)
    config["use_code"] = use_code

    assert use_code, "use_code cannot be empty; for random encoding, just press Enter."

    if use_code and use_code != "random":
        config["use_code_random"] = (ask("Use random extra bytes? (y/n, default y):", default="y").lower() != "n")

        if config["use_code_random"]:
            config["code_random_len"] = ask(
                "Length of random extra bytes (default 2):",
                default=2,
                cast=int,
            )
    else:
        config["min_code_len"] = ask("Minimum code length (default 4):", default=4, cast=int)
        config["max_code_len"] = ask("Maximum code length (default 4):", default=4, cast=int)

    config["use_ac"] = (ask("Use AC automaton for fixed-length patterns? (y/n, default n):", default="n").lower() == "y")

    config_path = Path(path) / "mix_tokenizer_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f"\n✔ Config saved to: {config_path}")


class MixTokenizerBase:
    """Base mixin to combine with an existing tokenizer class.

    NOTE: This is a dummy base class intended to be used via multiple inheritance
    with a real tokenizer class (see ``get_mix_tokenizer`` below).
    """

    use_code: Optional[str] = None
    use_ac: bool = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        ZH_CHARS = {chr(i) for i in range(ZH_RANGE[0], ZH_RANGE[1] + 1)}
        ZH_USED_CHARS = set()

        temp_list: List[List[int]] = []
        temp_ch: List[int] = []

        # 0. get config and prepare
        script_dir = Path(pretrained_model_name_or_path) / instance.dir_name

        config_path = script_dir / "mix_tokenizer_config.yaml"
        params = {}
        if not config_path.exists():
            create_config(pretrained_model_name_or_path)
        with config_path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        print(f"[INFO] Loaded MixTokenizer config from {config_path}: {params}")

        # 1. collect all Chinese characters from the dataset
        for ch in tqdm(ZH_CHARS | EXTRA_ZH_CHARS, desc="Collecting Chinese characters"):
            if isinstance(ch, int):
                ch = chr(ch)
            if params.get("use_code", "random") != "random":
                try:
                    ch.encode(params.get("use_code"))
                except Exception:
                    continue

            if ch in ZH_USED_CHARS:
                continue
            # store as ord to match downstream usage
            ZH_USED_CHARS.add(ch if isinstance(ch, int) else ord(ch))

        ZH_USED_CHARS = list(ZH_USED_CHARS)

        # random or some deterministic mapping
        instance.use_map = {}

        # 2. generate mapping patterns
        if params.get("use_code", "random") == "random":
            # random mapping
            print("[INFO] Generating random code mappings for Chinese characters.")
            alphabet = "".join(instance.byte_encoder.values())

            mapped_file = script_dir / "encode.txt"
            _flag = False
            if mapped_file.exists():
                print(f"[INFO] Loading existing code mappings from {mapped_file}.")
                with mapped_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        code, ch = line.strip().split("\t")
                        temp_list.append([ord(b) for b in code])
                        temp_ch.append(ord(ch))
                max_len = max(len(x) for x in temp_list)
                min_len = min(len(x) for x in temp_list)
                if len(temp_ch) == len(list(ZH_USED_CHARS)) and max_len == params.get("max_code_len", 5) and min_len == params.get("min_code_len", 2):
                    _flag = True
                else:
                    _flag = False
                print(f"[INFO] Loaded {len(temp_ch)} mappings. Code length range: {min_len} to {max_len}. flag={_flag}")

            if not _flag:
                print(f"[INFO] Generating new code mappings and saving to {mapped_file}.")
                print(f"[INFO] Total Chinese characters to map: {len(ZH_USED_CHARS)}")
                print(f"[INFO] Generating codes with length from {params.get('min_code_len', 2)} to {params.get('max_code_len', 5)}")
                temp_list = generate_non_substring_sequences_double_hash(
                    alphabet,
                    k=len(ZH_USED_CHARS),
                    min_len=params.get("min_code_len", 2),
                    max_len=params.get("max_code_len", 5),
                    seed=42,
                )
                # save to file
                temp_ch = list(ZH_USED_CHARS)
                script_dir.mkdir(parents=True, exist_ok=True)
                with mapped_file.open("w", encoding="utf-8") as f:
                    for code, ch in zip(temp_list, temp_ch):
                        f.write(f'{"".join(chr(b) for b in code)}\t{chr(ch)}\n')
                print(f"[INFO] Saved generated code mappings to {mapped_file}.")

        else:
            assert isinstance(params.get("use_code", None), str)
            print(f"[INFO] Using predefined code file: {params.get('use_code')}")
            temp_ch = []
            temp_rd: List[List[int]] = []
            if params.get("use_code_random", True):
                script_dir = Path(pretrained_model_name_or_path) / instance.dir_name
                rd_file = script_dir / f"code_random_{params.get('use_code')}.txt"
                _flag = False
                if rd_file.exists():
                    print(f"[INFO] Loading existing random code mappings from {rd_file}.")
                    with rd_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            temp_rd.append([ord(b) for b in line.strip()])
                    if len(temp_rd) == len(list(ZH_USED_CHARS)) and all(len(x) == params.get("code_random_len", 2) for x in temp_rd):
                        _flag = True
                    print(f"[INFO] Loaded {len(temp_rd)} random mappings. flag={_flag} extra_len={len(temp_rd[0]) if temp_rd else 0}")
                if not _flag:
                    print(f"[INFO] Generating new random code mappings and saving to {rd_file}.")
                    print(f"[INFO] Total Chinese characters to map: {len(ZH_USED_CHARS)}")
                    print(f"[INFO] Generating random codes with length {params.get('code_random_len', 2)}")
                    alphabet = "".join(instance.byte_encoder.values())
                    temp_rd = [random.choices(alphabet, k=params.get("code_random_len", 2)) for _ in range(len(ZH_USED_CHARS))]
                    with rd_file.open("w", encoding="utf-8") as f:
                        for code in temp_rd:
                            f.write(f'{"".join(code)}\n')
                    temp_rd = [[ord(b) for b in code] for code in temp_rd]

            for rd, ch in zip(temp_rd, ZH_USED_CHARS):
                code = chr(ch).encode(params.get("use_code"))
                temp_list.append(rd + [ord(instance.byte_encoder[b]) for b in code])
                temp_ch.append(ch)

        params["use_ac"] = params.get("use_ac") or not all(len(x) == len(temp_list[0]) for x in temp_list)

        # 3. build decoder
        if not params.get("use_ac"):
            print("[INFO] Using Hash Decoder for fixed-length patterns in MixTokenizer.")
            instance.mix_decoder = HybridDecoder(len(temp_list[0]))
            for pattern, ch in zip(temp_list, temp_ch):
                assert all(chr(x) in instance.byte_encoder.values() for x in pattern)
                instance.use_map[chr(ch)] = "".join(chr(b) for b in pattern)
                instance.mix_decoder.add_pattern(pattern, ch)
        else:
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
        instance.save_cache = load_from_folder(script_dir)
        return instance

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        bpe_tokens: List[str] = []
        for token in re.findall(self.pat, text):
            ret: List[str] = []
            for flag, l, r in self.splitter.py_split_zh_nonzh(token):
                if not flag:
                    t = "".join(self.byte_encoder[b] for b in token[l:r].encode("utf-8"))
                else:
                    parts = [self.use_map[ch] for ch in token[l:r]]
                    t = "".join(parts)
                ret.append(t)
            ret = "".join(ret)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(ret).split(" "))
        return bpe_tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens into a decoded string (optimized)."""
        tokens_str = "".join(tokens)
        ords = [ord(ch) for ch in tokens_str]

        byte_decoder = self.byte_decoder
        errors = self.errors

        utf8 = "utf-8"

        result_parts: List[str] = []

        intervals = self.mix_decoder.search(ords)
        for flag, l, r in intervals:
            if flag == -1:
                segment = tokens_str[l:r]
                decoded_bytes = bytearray(byte_decoder[ch] for ch in segment)
                result_parts.append(decoded_bytes.decode(utf8, errors=errors))
            else:
                result_parts.append(chr(flag))
        return "".join(result_parts)

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

def get_mix_tokenizer(tokenizer_cls, dir_name: str = "mix"):
    class_name = "MixTokenizer"
    mix_class = type(
        class_name,
        (MixTokenizerBase, tokenizer_cls),
        {"__module__": __name__, "dir_name": dir_name},
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
