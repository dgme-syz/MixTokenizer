import json
from typing import List

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from scipy.stats import qmc


def sample_integer_points(L: int, K: int, N: int, seed: int = 42) -> np.ndarray:
    """
    Sample N integer points within [0, L-1]^K using a quasi-random Sobol sequence.

    Args:
        L (int): Range of each dimension [0, L-1].
        K (int): Number of dimensions.
        N (int): Number of points to sample.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: (N, K) integer array of sampled points.
    """
    if seed is not None:
        np.random.seed(seed)

    # Sobol sequence produces low-discrepancy, quasi-uniform samples
    sampler = qmc.Sobol(d=K, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(N)))  # Sobol requires 2^m samples
    sobol_points = sampler.random_base2(m=m)[:N]

    int_points = np.floor(sobol_points * L).astype(int)
    np.random.shuffle(int_points)
    return int_points


class NewLangTokenizer:
    """
    Wrapper around BertWordPieceTokenizer for a custom language or symbol set.
    """

    def __init__(self, vocab_file: str | dict, **kwargs) -> None:
        if isinstance(vocab_file, str):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
        else:
            vocab = vocab_file
        for mark in ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]:
            if mark not in vocab:
                vocab[mark] = len(vocab)
            
        self.tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated") #important

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def tokenize(self, text: str) -> List[str]:
        x = self.tokenizer.encode(text, add_special_tokens=False).tokens
        return x

    def is_new_char(self, ch: str) -> bool:
        # Only treat single-character entries in vocab as new characters
        return len(ch) == 1 and ch in self.tokenizer.get_vocab()

    def decode(self, token_ids: List[int]) -> str:
        # Decode and remove spaces introduced by WordPiece
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
