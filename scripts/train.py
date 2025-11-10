import os
import re
import json

def main():
    import argparse
    parser = argparse.ArgumentParser("install scripts for vocab tokenizer")
    parser.add_argument("--output_dir", default="mix", type=str, help="path to save script.")    
    parser.add_argument("--model", type=str, help="path to your local model.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir = output_dir.rstrip("/\\")  
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load original tokenizer class
    tokenizer_config_path = os.path.join(args.model, "tokenizer_config.json")
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    tokenizer_cls_name = tokenizer_config.get("tokenizer_class")
    if not tokenizer_cls_name:
        raise ValueError("tokenizer_class not found in tokenizer_config.json")

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
        f'globals()["MixTokenizer"] = get_mix_tokenizer(tokenizer_cls=tokenizer_cls, dir_name=dir_name)'
    )

    new_tokenizer_code, n = re.subn(pattern, replacement, tokenizer_code)
    if n == 0:
        raise RuntimeError(f"Pattern {pattern} not found in tokenizer.py — check file structure")

    pattern = r'dir_name = "mix"'

    replacement = f'dir_name = "{os.path.basename(output_dir)}"'
    new_tokenizer_code, n = re.subn(pattern, replacement, new_tokenizer_code)
    if n == 0:
        raise RuntimeError(f"Pattern {pattern} not found in tokenizer.py — check file structure")

    updated_tokenizer_path = os.path.join(output_dir, "tokenizer.py")
    with open(updated_tokenizer_path, "w", encoding="utf-8") as f:
        f.write(new_tokenizer_code)
    print(f"✨ Saved updated tokenizer.py to {updated_tokenizer_path}")


if __name__ == "__main__":
    main()
