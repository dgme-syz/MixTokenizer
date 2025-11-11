# a script for converting a tokenizer into a mix tokenizer

import os
import json
from typing import List
from collections import Counter

import torch

from MixTokenizer import sample_integer_points

def get_mix_tokenizer(tokenizer_cls, expand: int = 0, expand_only: bool = False):

    class MixTokenizer(tokenizer_cls):
        """
        Combines Qwen2Tokenizer with an additional tokenizer for a custom language.
        Allows mapping of new language tokens to composite representations.
        """
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, new_lang_tokenizer, **kwargs):
            instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

            instance.new_lang_tokenizer = new_lang_tokenizer
            return instance
        
        def __len__(self):
            return super().__len__() + expand
        
        def save_to_json(self, output_dir: str = None):
            if output_dir is None:
                raise ValueError("Please provide 'output_dir' to save the mapping.")
            json_path = os.path.join(output_dir, "extra_config.json")
            cfg = {}
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            cfg["mapping"] = self.mapping
            cfg["used_ids"] = self.zero_ids
            cfg["level"] = self.level
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        def train(self, frequency_id_files: List[str] | Counter, level: int | str = 2, output_dir: str = None, seed: int = 42):
            """
            Train the MixTokenizer by preparing the mapping based on token frequencies.

            Args:
                frequency_id_files: List of file paths containing frequency data or a Counter object.
                level: The number of base tokens to combine for each new language token.
            """
            self._prepare_mapping(frequency_id_files, level, seed)
            self.save_to_json(output_dir=output_dir)

        def _prepare_mapping(self, frequency_id_files: List[str] | Counter, level: int | str, seed: int) -> None:
            """
            Build mapping between new language tokens and low-frequency Qwen tokens.
            """
            self.level = level
            if level == "mixed":
                raise NotImplementedError("Decoding in 'mixed' level mode is not supported.")

            # Aggregate frequency data
            frequency_counter = Counter()
            if isinstance(frequency_id_files, Counter):
                frequency_counter = frequency_id_files
            else:
                for file in frequency_id_files:
                    freq_data = torch.load(file)
                    frequency_counter.update(freq_data)

            # Ensure all base tokens are represented
            special_ids = set(self.all_special_ids)  # Qwen2Tokenizer 提供 .all_special_ids 属性

            # Ensure all base tokens are represented, but skip special tokens
            for i in range(len(self)):
                if i not in special_ids:
                    frequency_counter.update({i: 0})

            # Sort tokens by ascending frequency
            frequency_list = sorted(frequency_counter.items(), key=lambda x: (x[1], -x[0]))

            # Collect zero-frequency token IDs
            zero_ids = [tid for tid, freq in frequency_list if ((freq == 0 and tid not in special_ids) and (expand_only is False or tid >= len(self) - expand))]
            if not zero_ids:
                raise ValueError("No zero-frequency tokens available for mapping.")

            print(f"\033[91m[WORK]\033[0mFound {len(zero_ids)} zero-frequency tokens (not in special_ids) for mapping.")
            print(f"\033[91m[WORK]\033[0mNew language vocab size: {len(self.new_lang_tokenizer)}, use level={level}.")
            # Check available mapping capacity
            max_lim = int(pow(len(zero_ids), level))
            if max_lim < len(self.new_lang_tokenizer):
                raise ValueError(f"Increase 'level', max_lim = {max_lim}")

            # Randomly assign integer grid points for mapping
            points = sample_integer_points(L=len(zero_ids), K=level, N=len(self.new_lang_tokenizer), seed=seed)

            # Build mapping: new_lang_token_id → list[base_token_ids]
            self.mapping: List[list[int]] = [
                [zero_ids[x] for x in point] for point in points
            ]
            self.zero_ids = zero_ids
            self.zero_dict = {zid: idx for idx, zid in enumerate(zero_ids)}
            self.reverse_mapping = {point: idx for idx, point in enumerate(map(tuple, self.mapping))}

        
    return MixTokenizer
