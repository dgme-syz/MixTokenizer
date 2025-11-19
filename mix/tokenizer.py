# a script for converting a tokenizer into a mix tokenizer

from transformers import Qwen2Tokenizer

from MixTokenizer import get_mix_tokenizer


# ─────────── Model Directory Structure ───────────
#
# model/
# ├── tokenizer_config.json      # Hugging Face tokenizer configuration
# ├── model.safetensors         # Model weights
# ├── ...                       # Other model-related files
# └── mix/                     # Extra assets for MixTokenizer
#     ├── tokenizer.py          # Custom tokenizer wrapper
#
# ✨ Note: Keep the 'extra' folder intact to enable all MixTokenizer features!

dir_name = "mix"

# Get parent tokenizer's type
from transformers import Qwen2Tokenizer
tokenizer_cls = Qwen2Tokenizer

# Register dynamic class globally
globals()["MixTokenizer"] = get_mix_tokenizer(tokenizer_cls=tokenizer_cls, dir_name=dir_name)
