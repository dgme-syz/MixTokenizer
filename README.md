# MixTokenizer

Combine two tokenizers into one.

## Installation

```bash
pip install -e .

# For test
pip install -e .[dev]
pytest -s tests/
```

## Usage

Check and edit parameters in config.yaml, then run:

```bash
# Train and generate injection files (default output_dir: mix/)
python3 scripts/train.py --model /path/to/your/model/

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