import os
import re
import json
import yaml
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import transformers
from transformers import AutoTokenizer

from MixTokenizer.tokenizer_train import get_mix_tokenizer
from MixTokenizer.utils import NewLangTokenizer
from MixTokenizer.mapping import PrivateUnicodeMapper

def parse_args(config_file: str) -> argparse.Namespace:
    """Load configuration from YAML and convert to argparse.Namespace"""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def tokenize_file(file_path: str, mix_tokenizer_cls, model_path: str) -> Counter:
    """
    Tokenize all texts in a JSONL file and return a Counter of token IDs.
    Each process will call this function independently.
    """
    tokenizer_cls_name = json.load(open(os.path.join(model_path, "tokenizer_config.json"), "r"))["tokenizer_class"]
    tokenizer_cls = getattr(transformers, tokenizer_cls_name)
    raw_tokenizer = tokenizer_cls.from_pretrained(model_path)

    local_counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            text = example.get("text", "")
            token_ids = mix_tokenizer_cls.tokenize_texts([text], raw_tokenizer=raw_tokenizer)[0]
            local_counter.update(token_ids)
    return local_counter


def main():
    args = parse_args("config.yaml")
    print(args)
    output_dir = args.output_dir
    output_dir = output_dir.rstrip("/\\")  
    os.makedirs(output_dir, exist_ok=True)

    # 1. Mapping preparation
    if getattr(args.mapping, "mapping_file", None) and os.path.exists(args.mapping.mapping_file):
        mapper = PrivateUnicodeMapper.from_mapping_dict(mapping_file=args.mapping.mapping_file)
    else:
        old_areas_list = args.mapping.get("old_areas_list", [])
        seed = args.mapping.get("seed", 42)
        mapper = PrivateUnicodeMapper(old_areas_list=old_areas_list, seed=seed)

    mapping_dir = os.path.join(output_dir, "lang_map")
    os.makedirs(mapping_dir, exist_ok=True)
    mapper.save_mapping(os.path.join(mapping_dir, "mapping.json"))

    # 2. New language tokenizer
    tokenizer_type = args.new_lang_tokenizer.get("type", "default")
    tokenizer_path = args.new_lang_tokenizer.get("path")

    if tokenizer_type == "huggingface":
        new_lang_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    elif tokenizer_type == "default":
        vocab = mapper.get_vocab()
        new_lang_tokenizer = NewLangTokenizer(vocab)
    else:
        raise ValueError(f"Unsupported new language tokenizer type: {tokenizer_type}")
    # copy new_lang_tokenizer files to output_dir
    new_tokenizer_dir = os.path.join(output_dir, "new_tokenizer")
    os.makedirs(new_tokenizer_dir, exist_ok=True)
    new_lang_tokenizer.save_pretrained(new_tokenizer_dir)


    # 3. Load original tokenizer class
    tokenizer_config_path = os.path.join(args.model_name_or_path, "tokenizer_config.json")
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    tokenizer_cls_name = tokenizer_config.get("tokenizer_class")
    if not tokenizer_cls_name:
        raise ValueError("tokenizer_class not found in tokenizer_config.json")

    try:
        tokenizer_cls = getattr(transformers, tokenizer_cls_name)
    except AttributeError:
        tokenizer_cls_name = "Qwen2Tokenizer"
        tokenizer_cls = getattr(transformers, tokenizer_cls_name)
    mix_trained = args.mix_trained
    mix_tokenizer_cls = get_mix_tokenizer(tokenizer_cls=tokenizer_cls, expand=mix_trained.get("expand", 0), expand_only=mix_trained.get("expand_only", False))

    # 4. Frequency counting (multi-process if doc_path provided)
    counter = Counter()

    if mix_trained.get("doc_path", None):
        # JSONL files provided
        train_files = mix_trained["doc_path"]
        all_counters = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(tokenize_file, file, mix_tokenizer_cls, args.model_name_or_path)
                       for file in train_files]

            for future in as_completed(futures):
                all_counters.append(future.result())

        for c in all_counters:
            counter.update(c)
    elif mix_trained.get("counter_path", None):
        # Precomputed counter files
        for file in mix_trained["counter_path"]:
            if file.endswith(".json"):
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    counter.update({int(k): v for k, v in data.items()})
            elif file.endswith(".bin"):
                import torch
                freq_data = torch.load(file)
                counter.update(freq_data)
            else:
                raise ValueError(f"Unsupported counter file format: {file}")
    else:
        raise ValueError("Either 'mix_trained.doc_path' or 'mix_trained.counter_path' must be provided.")

    # 5. Train mix tokenizer
    mix_tokenizer = mix_tokenizer_cls.from_pretrained(
        args.model_name_or_path,
        new_lang_tokenizer=new_lang_tokenizer
    )
    mix_tokenizer.train(
        frequency_id_files=counter,
        level=mix_trained.get("level", 2),
        output_dir=output_dir,
        seed=mix_trained.get("seed", 42)
    )

    # 6. Update tokenizer.py with correct tokenizer class
    tokenizer_py_path = os.path.join("MixTokenizer", "tokenizer.py")
    with open(tokenizer_py_path, "r", encoding="utf-8") as f:
        tokenizer_code = f.read()

    pattern = (
        r"# Get parent tokenizer's type[\s\S]*?globals\(\)\[\"MixTokenizer\"\]\s*=\s*get_mix_tokenizer\(tokenizer_cls=tokenizer_cls, dir_name=dir_name\)"
    )
    replacement = (
        f"# Get parent tokenizer's type\n"
        f"from transformers import {tokenizer_cls_name}\n"
        f"tokenizer_cls = {tokenizer_cls_name}\n\n"
        f"# Register dynamic class globally\n"
        f'globals()["MixTokenizer"] = get_mix_tokenizer(tokenizer_cls=tokenizer_cls)'
    )

    new_tokenizer_code, n = re.subn(pattern, replacement, tokenizer_code)
    if n == 0:
        raise RuntimeError(f"Pattern {pattern} not found in tokenizer.py — check file structure")

    pattern = r'dir_name = "mix"'

    replacement = f'dir_name = "{os.path.basename(output_dir)}"'
    new_tokenizer_code, n = re.subn(pattern, replacement, new_tokenizer_code)
    if n == 0:
        raise RuntimeError(f"Pattern {pattern} not found in tokenizer.py — check file structure")

    os.makedirs(output_dir, exist_ok=True)
    updated_tokenizer_path = os.path.join(output_dir, "tokenizer.py")
    with open(updated_tokenizer_path, "w", encoding="utf-8") as f:
        f.write(new_tokenizer_code)
    print(f"✨ Saved updated tokenizer.py to {updated_tokenizer_path}")


if __name__ == "__main__":
    main()
