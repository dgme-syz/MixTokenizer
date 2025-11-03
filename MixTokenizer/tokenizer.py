# a script for converting a tokenizer into a mix tokenizer

import os
import json
from typing import List, Union

from transformers import AutoTokenizer

from MixTokenizer import NewLangTokenizer


# ─────────── Model Directory Structure ───────────
#
# model/
# ├── tokenizer_config.json      # Hugging Face tokenizer configuration
# ├── model.safetensors         # Model weights
# ├── ...                       # Other model-related files
# └── mix/                     # Extra assets for MixTokenizer
#     ├── extra_config.json     # Mapping info, level, frequency info
#     ├── tokenizer.py          # Custom tokenizer wrapper
#     ├── new_tokenizer/        # Your special language tokenizer (e.g., vocab.json)
#
# ✨ Note: Keep the 'extra' folder intact to enable all MixTokenizer features!


def get_mix_tokenizer(tokenizer_cls):

    class MixTokenizer(tokenizer_cls):
        """
        Combines Qwen2Tokenizer with an additional tokenizer for a custom language.
        Allows mapping of new language tokens to composite representations.
        """
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            instance.pretrained_model_name_or_path = pretrained_model_name_or_path
            # Load extra_config.json
            dir_name = "mix"
            script_dir = os.path.join(pretrained_model_name_or_path, dir_name)
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


            instance.new_lang_tokenizer = new_lang_tokenizer
            level = extra_config.get("level", None)

            if extra_config.get("mapping") and extra_config.get("used_ids"):
                print(f"Mapping file and used ids are loaded, level ignored: {level}")
                instance.mapping = extra_config["mapping"]
                instance.level = len(instance.mapping[0])
                instance.zero_ids = extra_config["used_ids"]
                instance.zero_dict = {zid: idx for idx, zid in enumerate(instance.zero_ids)}
                instance.reverse_mapping = {tuple(point): idx for idx, point in enumerate(instance.mapping)}
            else:
                raise ValueError(
                    "Ensure mapping and used_ids exist in config, or frequency_id_files and level exist in config."
                )
            return instance
        
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


# Get parent tokenizer's type
from transformers import Qwen2Tokenizer
tokenizer_cls = Qwen2Tokenizer

# Register dynamic class globally
globals()["MixTokenizer"] = get_mix_tokenizer(tokenizer_cls=tokenizer_cls)