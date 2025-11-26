import os
import json
import argparse
import shutil
import re


def load_tokenizer_config(config_path):
    """Load tokenizer_config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tokenizer_config(config_path, data):
    """Save updated tokenizer_config.json."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_inject_folder(auto_map_string):
    """
    Extract inject folder name from:
        "mix_Rw3Q5g/tokenizer.MixTokenizer"
    Returns:
        "mix_Rw3Q5g"
    """
    if not isinstance(auto_map_string, str):
        return None
    return auto_map_string.split("/")[0]


def update_tokenizer_py(py_path, new_inject_name):
    """Replace dir_name = "xxx" inside tokenizer.py."""
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = re.sub(
        r'dir_name\s*=\s*"(.*?)"',
        f'dir_name = "{new_inject_name}"',
        content
    )

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main(root_dir, new_inject_name=None):
    tokenizer_config_path = os.path.join(root_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        print(f"❌ tokenizer_config.json not found: {tokenizer_config_path}")
        return

    data = load_tokenizer_config(tokenizer_config_path)

    # Read auto_map.AutoTokenizer
    try:
        auto_tok_entry = data["auto_map"]["AutoTokenizer"][0]
    except Exception:
        print("❌ Missing auto_map.AutoTokenizer entry in tokenizer_config.json!")
        return

    old_inject = extract_inject_folder(auto_tok_entry)

    if old_inject:
        print(f"Current inject folder: {old_inject}")
    else:
        print("❌ Failed to parse inject folder name. Check tokenizer_config.json.")
        return

    if not new_inject_name:
        print("No new inject name specified. Reporting only, no modification.")
        return

    # --------------- Update tokenizer_config.json ---------------
    new_auto_map_str = f"{new_inject_name}/tokenizer.MixTokenizer"
    data["auto_map"]["AutoTokenizer"][0] = new_auto_map_str
    save_tokenizer_config(tokenizer_config_path, data)
    print(f"✔ Updated tokenizer_config.json: {new_auto_map_str}")

    # --------------- Update tokenizer.py ------------------------
    old_dir = os.path.join(root_dir, old_inject)
    if not os.path.exists(old_dir):
        print(f"❌ Inject folder does not exist: {old_dir}")
        return

    tokenizer_py_path = os.path.join(old_dir, "tokenizer.py")
    if not os.path.exists(tokenizer_py_path):
        print(f"❌ tokenizer.py not found: {tokenizer_py_path}")
        return

    update_tokenizer_py(tokenizer_py_path, new_inject_name)
    print(f"✔ Updated dir_name inside: {tokenizer_py_path}")

    # --------------- Rename inject folder -----------------------
    new_dir = os.path.join(root_dir, new_inject_name)
    shutil.move(old_dir, new_dir)
    print(f"✔ Renamed folder {old_inject} → {new_inject_name}")

    print("🎉 All tasks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject Folder Renamer")
    parser.add_argument("dir", type=str, help="Path to model directory")
    parser.add_argument("--new", type=str, default=None,
                        help="New inject folder name (optional)")
    args = parser.parse_args()

    main(args.dir, args.new)
