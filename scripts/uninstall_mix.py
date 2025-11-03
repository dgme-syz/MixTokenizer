import os
import shutil
from pathlib import Path

def remove_mix_folder(model_dir: Path, mix_folder_name="mix"):
    """
    Remove the 'mix' folder from the model directory.
    """
    mix_dir = model_dir / mix_folder_name
    if mix_dir.exists():
        shutil.rmtree(mix_dir)
        print(f"🗑️ Removed folder: {mix_dir}")
        return True
    else:
        print(f"⚠️ No mix folder found at {mix_dir}, skipping.")
        return False

def restore_tokenizer_config(model_dir: Path):
    """
    Restore tokenizer_config.json from tokenizer_config.json.bak
    """
    config_path = model_dir / "tokenizer_config.json"
    backup_path = model_dir / "tokenizer_config.json.bak"

    if not backup_path.exists():
        print("⚠️ Backup file tokenizer_config.json.bak not found, skipping restore.")
        return False

    shutil.move(str(backup_path), str(config_path))
    print(f"✅ Restored tokenizer_config.json from backup.")
    return True

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Uninstall MixTokenizer from a model directory")
    parser.add_argument("model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--mix_dir_name", type=str, default="mix", help="Name of the 'mix' folder to remove")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"❌ Model directory does not exist: {model_dir}")
        return

    print(f"⚙️ Restoring model directory: {model_dir}")
    confirm = input("⚠️ Confirm uninstall MixTokenizer? (y/N): ").strip().lower()
    if confirm != "y":
        print("❎ Operation cancelled.")
        return

    # Remove mix folder
    removed = remove_mix_folder(model_dir, args.mix_dir_name)
    # Restore tokenizer_config.json
    restored = restore_tokenizer_config(model_dir)

    if removed or restored:
        print("🎉 MixTokenizer uninstalled successfully.")
    else:
        print("ℹ️ Nothing to restore, model directory is already original.")

if __name__ == "__main__":
    main()
