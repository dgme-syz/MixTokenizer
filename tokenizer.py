# a script for converting a tokenizer into a mix tokenizer

import os
import json
from typing import List, Union
from collections import Counter

import torch
import transformers
from transformers import AutoTokenizer

from utils import sample_integer_points, NewLangTokenizer


# ─────────── Model Directory Structure ───────────
#
# model/
# ├── tokenizer_config.json      # Hugging Face tokenizer configuration
# ├── model.safetensors         # Model weights
# ├── ...                       # Other model-related files
# └── MixTokenizer/                     # Extra assets for MixTokenizer
#     ├── extra_config.json     # Mapping info, level, frequency info
#     ├── tokenizer.py          # Custom tokenizer wrapper
#     ├── utils.py              # Utility functions (e.g., sample_integer_points, NewLangTokenizer)
#     ├── new_tokenizer/        # Your special language tokenizer (e.g., vocab.json)
#     └── lang_map/             # Mapping: new language chars → old language chars
#
# ✨ Note: Keep the 'extra' folder intact to enable all MixTokenizer features!


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)   


def get_mix_tokenizer(
    tokenizer_cls, 
    new_lang_tokenizer,
    extra_config,
):

    class MixTokenizer(tokenizer_cls):
        """
        Combines Qwen2Tokenizer with an additional tokenizer for a custom language.
        Allows mapping of new language tokens to composite representations.
        """
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            instance.new_lang_tokenizer = new_lang_tokenizer
            level = extra_config.get("level", None)
            frequency_id_files = extra_config.get("frequency_id_files", None)

            if extra_config.get("mapping") and extra_config.get("used_ids"):
                print(f"Mapping file and used ids are loaded, level ignored: {level}")
                instance.mapping = extra_config["mapping"]
                instance.level = len(instance.mapping[0])
                instance.zero_ids = extra_config["used_ids"]
                instance.zero_dict = {zid: idx for idx, zid in enumerate(instance.zero_ids)}
                instance.reverse_mapping = {tuple(point): idx for idx, point in enumerate(instance.mapping)}
            elif frequency_id_files:
                print(f"Mapping file and used ids do not all exist, using frequency_id_files, level={level}")
                instance._prepare_mapping(frequency_id_files, level)
                instance.save_to_json()
            else:
                raise ValueError(
                    "Ensure mapping and used_ids exist in config, or frequency_id_files and level exist in config."
                )
            return instance
        
        def save_to_json(self):
            json_path = os.path.join(script_dir, "extra_config.json")
            cfg = {}
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            if not cfg.get("mapping") or not cfg.get("used_ids"):
                cfg["mapping"] = self.mapping
                cfg["used_ids"] = self.zero_ids
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
                
        def _prepare_mapping(self, frequency_id_files: List[str], level: int | str) -> None:
            """
            Build mapping between new language tokens and low-frequency Qwen tokens.
            """
            if level == "mixed":
                raise NotImplementedError("Decoding in 'mixed' level mode is not supported.")

            # Aggregate frequency data
            frequency_counter = Counter()
            for file in frequency_id_files:
                freq_data = torch.load(file)
                frequency_counter.update(freq_data)

            # Ensure all base tokens are represented
            for i in range(len(self)):
                frequency_counter.update({i: 0})

            # Sort tokens by ascending frequency
            frequency_list = sorted(frequency_counter.items(), key=lambda x: x[1])

            # Collect zero-frequency token IDs
            zero_ids = [tid for tid, freq in frequency_list if freq == 0]
            if not zero_ids:
                raise ValueError("No zero-frequency tokens available for mapping.")

            print(f"\033[91m[WORK]\033[0mFound {len(zero_ids)} zero-frequency tokens for mapping.")
            print(f"\033[91m[WORK]\033[0mNew language vocab size: {len(self.new_lang_tokenizer)}, use level={level}.")
            # Check available mapping capacity
            max_lim = int(pow(len(zero_ids), level))
            if max_lim < len(self.new_lang_tokenizer):
                raise ValueError(f"Increase 'level', max_lim = {max_lim}")

            # Randomly assign integer grid points for mapping
            points = sample_integer_points(L=len(zero_ids), K=level, N=len(self.new_lang_tokenizer))

            # Build mapping: new_lang_token_id → list[base_token_ids]
            self.mapping: List[list[int]] = [
                [zero_ids[x] for x in point] for point in points
            ]
            self.zero_ids = zero_ids
            self.zero_dict = {zid: idx for idx, zid in enumerate(zero_ids)}
            self.reverse_mapping = {point: idx for idx, point in enumerate(map(tuple, self.mapping))}

        def _same_char_type(self, ch1: str, ch2: str) -> bool:
            return self.new_lang_tokenizer.is_new_char(ch1) == self.new_lang_tokenizer.is_new_char(ch2)

        def _same_id_type(self, id1: int, id2: int) -> bool:
            return self.is_new_id(id1) == self.is_new_id(id2)

        def is_new_id(self, token_id: int) -> bool:
            return token_id in self.zero_dict

        def _convert_ids_to_new_lang_ids(self, token_ids: List[int]) -> int | List[int]:
            return self.reverse_mapping.get(tuple(token_ids))

        def tokenize(self, text: str, **kwargs) -> List[str]:
            """
            Two-stage tokenization:
            1. Group characters by type (new_lang vs Qwen).
            2. Tokenize each segment using the appropriate tokenizer.
            """
            sub_texts = []
            for ch in text:
                if sub_texts and self._same_char_type(ch, sub_texts[-1][0][0]):
                    sub_texts[-1][0] += ch
                else:
                    sub_texts.append([ch, self.new_lang_tokenizer.is_new_char(ch)])

            tokens = []
            for sub_text, is_new in sub_texts:
                if not is_new:
                    tokens.extend(tokenizer_cls.tokenize(self, sub_text, **kwargs))
                else:
                    # print("**", sub_text)
                    tokens.extend(self.new_lang_tokenizer.tokenize(sub_text))
                    # print(tokens, len(self.new_lang_tokenizer))
            return tokens

        def _convert_one_token_to_id(self, token: str) -> Union[int, List[int]]:
            """
            Convert a token to one or more IDs depending on its type.
            """
            if self.new_lang_tokenizer.is_new_char(token):
                token_id = self.new_lang_tokenizer.tokenizer.token_to_id(token)
                # print("***||")
                # print(token_id, self.mapping[token_id])
                return self.mapping[token_id]
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

        def _decode(self, token_ids: Union[int, List[int]], **kwargs) -> str:
            """
            Decode a sequence of token IDs by segment type.
            """
            def _decode_sub(ids: List[int]) -> str:
                # print(ids, ">>>>>>>>>>>>>>>>")
                if not ids:
                    return ""
                if all(self.is_new_id(x) for x in ids):
                    assert len(ids) % self.level == 0, "Invalid new language token IDs length."
                    ids = [self._convert_ids_to_new_lang_ids(ids[i:i + self.level]) for i in range(0, len(ids), self.level)]
                    return self.new_lang_tokenizer.decode(ids)
                return tokenizer_cls._decode(self, ids, **kwargs)

            sub_text, buffer = "", []
            for tid in token_ids:
                if buffer and self._same_id_type(tid, buffer[-1]):
                    buffer.append(tid)
                else:
                    sub_text += _decode_sub(buffer)
                    buffer = [tid]
            sub_text += _decode_sub(buffer)
            return sub_text
        
    return MixTokenizer



# Load extra_config.json
json_path = os.path.join(script_dir, "extra_config.json")
with open(json_path, "r", encoding="utf-8") as f:
    extra_cfg = json.load(f)

# Load new tokenizer
new_path = os.path.join(script_dir, "new_tokenizer")
try:
    print("Try to load AutoTokenizer from HF")
    new_tokenizer = AutoTokenizer.from_pretrained(new_path)
except Exception:
    print("Fallback: use default WordLevel Tokenizer")
    vocab_path = os.path.join(new_path, "vocab.json")
    new_tokenizer = NewLangTokenizer(vocab_file=vocab_path)

# Get parent tokenizer's type
try:
    tokenizer_config_path = os.path.join(parent_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_cls_name = json.load(f)["tokenizer_class"]
    tokenizer_cls = getattr(transformers, tokenizer_cls_name)
except Exception:
    try:
        tokenizer = AutoTokenizer.from_pretrained(parent_dir)
        tokenizer_cls = tokenizer.__class__
    except Exception:
        from transformers import Qwen2Tokenizer
        tokenizer_cls = Qwen2Tokenizer

# Register dynamic class globally
globals()["MixTokenizer"] = get_mix_tokenizer(
    tokenizer_cls=tokenizer_cls, new_lang_tokenizer=new_tokenizer, extra_config=extra_cfg,
)

if __name__=='__main__':
    path = "/home/nfs06/shenyz/models/Qwen3-0.6B"
    MixTokenizerCls = globals()["MixTokenizer"]
    from lang_map.mapping import PrivateUnicodeMapper
    MAPPER = PrivateUnicodeMapper.from_mapping_dict(mapping_file="lang_map/mapping.json")
    mix_tokenizer = MixTokenizerCls.from_pretrained(path)
    raw_tokenizer = Qwen2Tokenizer.from_pretrained(path)
    print("\033[91m[DONE]\033[0mTokenizers prepared.")

    # Test 1. Ensure mapping and unmapping works
    if MAPPER.unmap_string(MAPPER.map_string("你好，世界！")) == "你好，世界！":
        print("="*20)
        print("Mapping and unmapping test passed.")
        print("="*20)
    
    # Test 2. mixtokenizer can decode and encode
    ## Test 2.1 source text keep unchanged
    zh = "你好，世界！"
    m_inputs1 = mix_tokenizer([zh], return_tensors="pt")
    m_inputs2 = raw_tokenizer([zh], return_tensors="pt")
    if torch.equal(m_inputs1["input_ids"], m_inputs2["input_ids"]):
        print("="*20)
        print("MixTokenizer and RawTokenizer input_ids match on original text.")
        print("="*20)

   ## Test 2.2 mixtokenizer can recover mapped text
    mapped_zh = MAPPER.map_string(zh)
    m_inputs_mapped = mix_tokenizer([mapped_zh], return_tensors="pt")
    raw_outputs = raw_tokenizer([mapped_zh], return_tensors="pt")
    decoded_mapped = mix_tokenizer.batch_decode(m_inputs_mapped["input_ids"], skip_special_tokens=True)[0]
    unmapped_decoded = MAPPER.unmap_string(decoded_mapped)
    print("="*20)
    print(
        f'Mapped input_ids:\n\n{m_inputs_mapped}\n\n'
        f'Decoded mapped text: {decoded_mapped}\n\n'
        f'(unicode) Decoded mapped text: {[hex(ord(c)) for c in decoded_mapped]}\n\n'
        f'Unmapped decoded text: {unmapped_decoded}\n'
        f'all raw encoded: {raw_outputs}\n'
    )
    if (
        unmapped_decoded == zh
        and not torch.equal(m_inputs_mapped["input_ids"], raw_outputs["input_ids"])
    ):
        print("MixTokenizer successfully recovers original text from mapped input.")
    print("="*20)

    ## Test 2.3 mixtokenizer can recover mix text
    mix_zh = "你好"+decoded_mapped+"！"
    m_inputs_mix = mix_tokenizer([mix_zh], return_tensors="pt")
    raw_outputs = raw_tokenizer([mix_zh], return_tensors="pt")
    decoded_mix = mix_tokenizer.batch_decode(m_inputs_mix["input_ids"], skip_special_tokens=True)[0]
    unmapped_decoded_mix = MAPPER.unmap_string(decoded_mix)
    print("="*20)
    print(
        f'Mix input_ids:\n\n{m_inputs_mix}\n\n'
        f'Decoded mix text: {decoded_mix}\n\n'
        f'(unicode) Decoded mix text: {[hex(ord(c)) for c in decoded_mix]}\n\n'
        f'Unmapped decoded mix text: {unmapped_decoded_mix}\n'
        f'all raw encoded: {raw_outputs}\n'
    )
    if (
        unmapped_decoded_mix == "你好"+zh+"！"
        and not torch.equal(m_inputs_mix["input_ids"], raw_outputs["input_ids"])
    ):
        print("MixTokenizer successfully recovers original text from mixed input.")
    print("="*20)
