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

just run:

```bash
# Train and generate injection files (default output_dir: mix/)
python3 scripts/train.py --model /path/to/your/model/

# Inject MixTokenizer into your model
python3 scripts/install_mix.py /path/to/your/model/ --mix_dir mix/

# Rename inject dir
python3 scripts/rename.py /path/to/your/model/ --new new_inject_dir_name

# Uninstall MixTokenizer and restore original files
python3 scripts/uninstall_mix.py /path/to/your/model/ --mix_dir_name mix

```
## 🛠️ Configuration

`train.py` includes an interactive configuration generator for MixTokenizer.  
It guides you through creating a `mix_tokenizer_config.yaml` file that controls:

- Encoding strategy  
- Code-length rules  
- AC-automaton optimizations for fast pattern matching  

---

### ⚙️ Available Options

#### **`use_code`**
- Encoding to use (e.g., `gbk`).
- Leave empty to enable **random-code mode**.

#### **When `use_code` *is provided***:
- **`use_code_random`**  
  Whether to append extra random bytes (default: `y`).

- **`code_random_len`**  
  Length of the appended random bytes (default: `2`).

#### **When `use_code` *is empty*** (Random-code mode):
- **`min_code_len`**  
  Minimum generated code length (default: `4`).

- **`max_code_len`**  
  Maximum generated code length (default: `4`).

#### **`use_ac`**
- Enables AC automaton optimization for fast matching of fixed-length patterns.  
- Default: off (`n`).

### 💎 Example

```yaml
max_code_len: 4
min_code_len: 4
use_ac: false
use_code: null
```

## Load

Before uninstalling, you can load the model as usual:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model/", trust_remote_code=True)
```