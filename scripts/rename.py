import os
import json
import argparse
import shutil
import re


def load_tokenizer_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tokenizer_config(config_path, data):
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_inject_folder(auto_map_string):
    """
    输入: "mix_Rw3Q5g/tokenizer.MixTokenizer"
    返回: "mix_Rw3Q5g"
    """
    if not isinstance(auto_map_string, str):
        return None
    return auto_map_string.split("/")[0]


def update_tokenizer_py(py_path, new_inject_name):
    """替换 tokenizer.py 中 dir_name = "..."""
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
        print(f"❌ 找不到 tokenizer_config.json: {tokenizer_config_path}")
        return

    data = load_tokenizer_config(tokenizer_config_path)

    # 读取 auto_map
    try:
        auto_tok_entry = data["auto_map"]["AutoTokenizer"][0]
    except Exception:
        print("❌ tokenizer_config.json 中缺少 auto_map.AutoTokenizer 信息！")
        return

    old_inject = extract_inject_folder(auto_tok_entry)

    if old_inject:
        print(f"当前 Inject 文件夹名：{old_inject}")
    else:
        print("❌ 无法解析 Inject 文件夹名，检查 tokenizer_config.json")
        return

    if not new_inject_name:
        print("未指定新的 Inject 名称：仅报告，不修改。")
        return

    # --------------- 修改 tokenizer_config.json ---------------
    new_auto_map_str = f"{new_inject_name}/tokenizer.MixTokenizer"
    data["auto_map"]["AutoTokenizer"][0] = new_auto_map_str
    save_tokenizer_config(tokenizer_config_path, data)
    print(f"✔ 已更新 tokenizer_config.json: {new_auto_map_str}")

    # --------------- 修改 tokenizer.py -------------------------
    old_dir = os.path.join(root_dir, old_inject)
    if not os.path.exists(old_dir):
        print(f"❌ 目录不存在：{old_dir}")
        return

    tokenizer_py_path = os.path.join(old_dir, "tokenizer.py")
    if not os.path.exists(tokenizer_py_path):
        print(f"❌ 未找到文件：{tokenizer_py_path}")
        return

    update_tokenizer_py(tokenizer_py_path, new_inject_name)
    print(f"✔ 已修改 {tokenizer_py_path} 内 dir_name 字段")

    # --------------- 重命名文件夹 -----------------------------
    new_dir = os.path.join(root_dir, new_inject_name)
    shutil.move(old_dir, new_dir)
    print(f"✔ 已将目录 {old_inject} 重命名为 {new_inject_name}")

    print("🎉 全部步骤完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject Folder Renamer")
    parser.add_argument("dir", type=str, help="模型目录路径")
    parser.add_argument("--new", type=str, default=None,
                        help="新的 Inject 文件夹名字 (可选)")
    args = parser.parse_args()

    main(args.dir, args.new)
