#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch map / unmap JSONL documents using Private Unicode mapping.

Usage:
    python batch_map_umap.py --mapping mapping.json --inputs data1.jsonl data2.jsonl --output_dir mapped --mode map --num_workers 8
    python batch_map_umap.py --mapping mapping.json --inputs mapped/data1.jsonl --output_dir unmapped --mode umap
"""

import os
import json
import random
import argparse
from typing import List, Union, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor
from functools import partial


class PrivateUnicodeMapper:
    """
    Map characters from specified Unicode ranges to Private Use Areas (PUA, SPUA-A, SPUA-B)
    and support bi-directional string conversion.
    """

    def __init__(self, old_areas_list: List[Union[Tuple[int, int], int]], seed: int = 42):
        """
        Initialize the mapper and generate a random mapping.

        Args:
            old_areas_list: List of Unicode ranges (start, end) or single code points.
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.old_codes = self._collect_codes(old_areas_list)
        self.private_area = self._build_private_area()
        self._check_capacity()
        self.mapping, self.reverse_mapping = self._build_mapping()

    @classmethod
    def from_mapping_dict(cls, mapping_file: Union[Dict[str, int], str]):
        """Create an instance from an existing mapping dictionary or JSON file."""
        if isinstance(mapping_file, str):
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping_dict = json.load(f)
        else:
            mapping_dict = mapping_file

        mapping = {ord(k): v for k, v in mapping_dict.items()}
        reverse_mapping = {v: k for k, v in mapping.items()}
        instance = cls.__new__(cls)
        instance.mapping = mapping
        instance.reverse_mapping = reverse_mapping
        return instance

    def save_mapping(self, filepath: str):
        """Save the current mapping to a JSON file (old -> new)."""
        mapping_dict = {chr(k): v for k, v in self.mapping.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

    def get_vocab(self, save: bool = False) -> Dict[str, int]:
        """Build a vocabulary file for the mapped private characters."""
        private_chars = list(self.reverse_mapping.keys())
        vocab = {chr(c): i for i, c in enumerate(private_chars)}

        if save:
            with open("vocab.json", "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
        return vocab

    def _collect_codes(self, old_areas_list):
        """Collect all Unicode code points from ranges or single points."""
        codes = []
        for area in old_areas_list:
            if isinstance(area, list):
                start, end = area
                codes.extend(range(start, end + 1))
            elif isinstance(area, int):
                codes.append(area)
            else:
                raise ValueError(f"Invalid area: {area}")
        return codes

    def _build_private_area(self):
        """Return list of all private area code points (PUA + SPUA-A + SPUA-B)."""
        pua_basic = range(0xE000, 0xF8FF + 1)
        pua_a = range(0xF0000, 0xFFFFD + 1)
        pua_b = range(0x100000, 0x10FFFD + 1)
        return list(pua_basic) + list(pua_a) + list(pua_b)

    def _check_capacity(self):
        """Ensure there are enough private area slots for all original code points."""
        if len(self.old_codes) > len(self.private_area):
            raise ValueError(
                f"Number of original code points ({len(self.old_codes)}) exceeds available private area "
                f"({len(self.private_area)}). Reduce old_areas or split mapping."
            )

    def _build_mapping(self):
        """Randomly assign old codes to private area codes."""
        random.seed(self.seed)
        mapped_private = random.sample(self.private_area, len(self.old_codes))
        mapping = {orig: priv for orig, priv in zip(self.old_codes, mapped_private)}
        reverse_mapping = {v: k for k, v in mapping.items()}
        return mapping, reverse_mapping

    def map_string(self, text: str) -> str:
        """Map a string’s characters to private area characters."""
        return "".join(chr(self.mapping.get(ord(c), ord(c))) for c in text)

    def unmap_string(self, text: str) -> str:
        """Unmap a private area string back to its original characters."""
        return "".join(chr(self.reverse_mapping.get(ord(c), ord(c))) for c in text)

    def _map_text_line(self, line: str) -> str:
        """Map the 'text' field of a single JSONL line."""
        data = json.loads(line)
        if "text" in data:
            data["text"] = self.map_string(data["text"])
        return json.dumps(data, ensure_ascii=False)

    def _unmap_text_line(self, line: str) -> str:
        """Unmap the 'text' field of a single JSONL line."""
        data = json.loads(line)
        if "text" in data:
            data["text"] = self.unmap_string(data["text"])
        return json.dumps(data, ensure_ascii=False)

    def map_docs_file(self, input_path: str, output_path: str, num_workers: int = 4):
        """Map all 'text' fields in a JSONL file to private Unicode using multiprocessing."""
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        mapper_func = partial(PrivateUnicodeMapper._map_text_line, self)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            mapped_lines = list(executor.map(mapper_func, lines))

        with open(output_path, "w", encoding="utf-8") as f:
            for line in mapped_lines:
                f.write(line + "\n")

    def unmap_docs_file(self, input_path: str, output_path: str, num_workers: int = 4):
        """Unmap all 'text' fields in a JSONL file back from private Unicode."""
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        unmapper_func = partial(PrivateUnicodeMapper._unmap_text_line, self)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            unmapped_lines = list(executor.map(unmapper_func, lines))

        with open(output_path, "w", encoding="utf-8") as f:
            for line in unmapped_lines:
                f.write(line + "\n")


# ---------- Batch processing logic ----------

def ensure_dir(path: str):
    """Create output directory if it doesn’t exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_single_file(mapper: PrivateUnicodeMapper, input_path: str, output_path: str, mode: str, num_workers: int):
    """Apply map or unmap to a single JSONL file."""
    if mode == "map":
        mapper.map_docs_file(input_path, output_path, num_workers=num_workers)
    elif mode == "umap":
        mapper.unmap_docs_file(input_path, output_path, num_workers=num_workers)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def process_files(mapper: PrivateUnicodeMapper, input_paths: List[str], output_dir: str, mode: str, num_workers: int):
    """Batch process multiple JSONL files."""
    ensure_dir(output_dir)
    for path in input_paths:
        basename = f"{mode}_" + os.path.basename(path)
        out_path = os.path.join(output_dir, basename)
        print(f"[{mode.upper()}] {path} -> {out_path}")
        process_single_file(mapper, path, out_path, mode, num_workers)
    print("✅ All files processed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Batch map / unmap JSONL documents")
    parser.add_argument("--mapping", required=True, help="Path to mapping.json file")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input file(s) or directory path(s)")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--mode", choices=["map", "umap"], required=True, help="Operation mode")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    mapper = PrivateUnicodeMapper.from_mapping_dict(mapping_file=args.mapping)

    input_files = []
    for item in args.inputs:
        if os.path.isdir(item):
            input_files.extend(
                [os.path.join(item, f) for f in os.listdir(item) if f.endswith(".jsonl")]
            )
        elif os.path.isfile(item):
            input_files.append(item)
        else:
            print(f"⚠️ Skipping invalid input: {item}")

    if not input_files:
        print("❌ No input files found.")
        return

    process_files(mapper, input_files, args.output_dir, args.mode, args.num_workers)


if __name__ == "__main__":
    main()
