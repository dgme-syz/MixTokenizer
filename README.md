# MixTokenizer

Combine two tokenizers into one.

## Installation

```bash
pip install -e .
```
## Mapping

```bash
# map zh -> new 
python MixTokenizer/lang_map/mapping.py --mapping MixTokenizer/lang_map/mapping.json --inputs data/text.jsonl --output_dir data/ --mode map

# umap new -> zh
python MixTokenizer/lang_map/mapping.py --mapping MixTokenizer/lang_map/mapping.json --inputs data/map_text.jsonl --output_dir data/ --mode umap
# inputs and outputs can be dir
```
Eg：
```markdown
src: 就是因为有这些人认同我……所以我才能够……无论是有妖狐在我的体内，或是被村子里的人们以冷漠的眼光看待，我都不觉得难过了……因为我……已经不是孤单一人了！

new: 􅊫􀞡󳉅󻶉􈽃􎾻󾓚􈯹󲆼􈻥􃬚􃬚􂓐􁛖􈻥󱐖󽉓󽐇􃬚􃬚􄼓󼑹􀞡􈽃􄝌󸌀󷉤􈻥􀳇󿠵󹔐􏂬􀞡󴼘󴍪􈐕􍙾􀳇􈯹􌼽􁛖􅲠󶡦􀳇􉨑􆥅󴷙󱧒􏂬􈻥󲝤󽔾􏽌􆍑󷫨󻾂󲉐􃬚􃬚󳉅󻶉􈻥􃬚􃬚􍓒󾜋󽔾􀞡􎩬󸓴󵤀􈯹󲉐󻢎

```


## Usage

Check and edit parameters in config.yaml, then run:

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