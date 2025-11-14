# MixTokenizer

Combine two tokenizers into one.

## Installation

```bash
pip install -e .

# For test
pip install -e .[dev]
pytest -s tests/
```
## Mapping

```bash
# map zh -> new 
python MixTokenizer/lang_map/mapping.py --mapping MixTokenizer/lang_map/mapping.json --inputs data/text.jsonl --output_dir data/ --mode map

# umap new -> zh
python MixTokenizer/lang_map/mapping.py --mapping MixTokenizer/lang_map/mapping.json --inputs data/map_text.jsonl --output_dir data/ --mode umap
# inputs and outputs can be dir

# map hf repo
python MixTokenizer/lang_map/mapping.py \
    --mapping MixTokenizer/lang_map/mapping.json \
    --inputs meta-math/GSM8K_zh \
    --hf_split train \
    --hf_text_fields question_zh answer_zh \
    --output_dir tmp \
    --mode map 
    # --push_to_hf username/mapped-dataset
```
Eg：
```markdown
src: 就是因为有这些人认同我……所以我才能够……无论是有妖狐在我的体内，或是被村子里的人们以冷漠的眼光看待，我都不觉得难过了……因为我……已经不是孤单一人了！

new: "\U001052ab\U001007a1\U000f3245\U000fbd89\U00108f43\U0010efbb\U000fe4da\U00108bf9\ue319\U000f21bc\U00108ee5\U00103b1a\U00103b1a\U001024d0\U001016d6\U00108ee5\U000f1416\U000fd253\U000fd407\U00103b1a\U00103b1a\U00104f13\U000fc479\U001007a1\U00108f43\U0010474c\U000f8300\U000f7264\U00108ee5\U00100cc7\U000ff835\U000f9510\U0010f0ac\uf32a\U001007a1\U000f4f18\U000f436a\U00108415\U0010d67e\U00100cc7\U00108bf9\U0010cf3d\U001016d6\U00105ca0\U000f6866\U00100cc7\U00109a11\U00106945\U000f4dd9\U000f19d2\U0010f0ac\U00108ee5\U000f2764\U000fd53e\U0010ff4c\U00106351\U000f7ae8\U000fbf82\U000f2250\U00103b1a\U00103b1a\U000f3245\U000fbd89\U00108ee5\U00103b1a\U00103b1a\U0010d4d2\U000fe70b\U000fd53e\U001007a1\U0010ea6c\U000f84f4\U000f5900\U00108bf9\U000f2250\U000fb88e"

```


## Usage

Check and edit parameters in config.yaml, then run:

For exanding vocabs, we provide `mix_trained.expand` (expand the vocabulary size) and `mix_trained.expand_only` (just use expand id or not) in `config.yaml`, you can see their usage in `scripts/train.py`.

```bash
# Train and generate injection files (default output_dir: mix/)
python3 scripts/train.py

# Inject MixTokenizer into your model
python3 scripts/install_mix.py /path/to/your/model/ --mix_dir mix/

# Uninstall MixTokenizer and restore original files
python3 scripts/uninstall_mix.py /path/to/your/model/ --mix_dir_name mix

```

## Load

Before uninstalling, you can load the model as usual:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model/", trust_remote_code=True)
```