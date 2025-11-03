import os
import shutil
import json
from pathlib import Path

def copy_mix_folder(target_model_dir: Path, source_mix_dir: Path):
    """
    Copy the 'mix' folder to the model directory.
    """
    dest_dir = target_model_dir / os.path.basename(source_mix_dir)
    if dest_dir.exists():
        print(f"🗑️ Removing existing folder at {dest_dir}")
        shutil.rmtree(dest_dir)
    shutil.copytree(source_mix_dir, dest_dir)
    print(f"✅ Copied '{source_mix_dir}' to '{dest_dir}'")
    return dest_dir

def update_tokenizer_config(model_dir: Path, mix_folder_name="mix"):
    """
    Backup tokenizer_config.json and update it:
    - tokenizer_class -> "MixTokenizer"
    - auto_map -> {"AutoTokenizer": ["mix/tokenizer.MixTokenizer", null]}
    """
    config_path = model_dir / "tokenizer_config.json"
    backup_path = model_dir / "tokenizer_config.json.bak"

    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist!")
    if not backup_path.exists():
        # Backup
        shutil.copy2(config_path, backup_path)
        print(f"💾 Backup created at {backup_path}")

        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Update
        config["tokenizer_class"] = "MixTokenizer"
        config["auto_map"] = {
            "AutoTokenizer": [f"{mix_folder_name}/tokenizer.MixTokenizer", None]
        }

        # Save
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✨ tokenizer_config.json updated successfully.")
    else:
        print(f"⚠️ Backup already exists at {backup_path}, skipping update.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Install MixTokenizer to a model directory")
    parser.add_argument("model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--mix_dir", type=str, default="./mix", help="Path to the 'mix' folder (relative to scripts)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    args.mix_dir = args.mix_dir.rstrip("/\\")
    mix_dir = Path(args.mix_dir).resolve()

    if not model_dir.exists():
        print(f"❌ Model directory does not exist: {model_dir}")
        return
    if not mix_dir.exists():
        print(f"❌ Mix folder does not exist: {mix_dir}")
        return

    # Copy mix folder
    copy_mix_folder(model_dir, mix_dir)

    # Update tokenizer_config.json
    update_tokenizer_config(model_dir, mix_folder_name=os.path.basename(mix_dir))

if __name__ == "__main__":
    main()
