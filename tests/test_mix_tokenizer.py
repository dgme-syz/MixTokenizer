import json
import os
import time
from typing import List
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer
from datasets import Dataset
import pickle
import tempfile

class NewLangTokenizer:
    """Wrapper around WordLevel tokenizer for a custom language or symbol set."""

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
        self.tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text, add_special_tokens=False).tokens

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")


@pytest.fixture(scope="module")
def user_model_paths():
    path1 = input("Enter path for mixtokenizer AutoTokenizer model: ").strip()
    path2 = input("Enter path for standard AutoTokenizer model: ").strip()
    return path1, path2


@pytest.fixture(scope="module")
def tokenizers(user_model_paths):
    path1, path2 = user_model_paths
    tok1 = AutoTokenizer.from_pretrained(path1, trust_remote_code=True)
    tok2 = AutoTokenizer.from_pretrained(path2)
    return tok1, tok2


@pytest.fixture
def sample_text():
    return ("􅊫􀞡󳉅󻶉􈽃􎾻󾓚􈯹" * 1000)


def test_tokenizer_speed(tokenizers, sample_text):
    tok1, tok2 = tokenizers
    x = {}
    for name, tok in zip(["Tokenizer1", "Tokenizer2"], [tok1, tok2]):
        start = time.perf_counter()
        tokens = tok([sample_text], return_tensors="pt")["input_ids"][0].tolist()
        end = time.perf_counter()
        x[name] = tokens
        print(f"{name} tokenized {len(sample_text)} chars into {len(tokens)} tokens in {end-start:.4f}s")
    assert len(x["Tokenizer1"]) > 0
    assert len(x["Tokenizer2"]) > 0
    print("Tokenizer speed test passed.")


def test_decoder_consistency(tokenizers, sample_text):
    tok1, tok2 = tokenizers
    tokens1 = tok1([sample_text], return_tensors="pt")["input_ids"][0].tolist()
    decoded = tok1.decode(tokens1)
    assert decoded == sample_text
    print("Decoder consistency test passed.")


def test_pickle_serialization(tokenizers):
    tok1, _ = tokenizers
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = os.path.join(tmpdir, "tok.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(tok1, f)
        with open(pkl_path, "rb") as f:
            loaded_tok = pickle.load(f)
        tokens = tok1(["test"], return_tensors="pt")["input_ids"][0].tolist()
        loaded_tokens = loaded_tok(["test"], return_tensors="pt")["input_ids"][0].tolist()
        assert tokens == loaded_tokens
    print("Pickle serialization test passed.")


def test_dataset_map(tokenizers, sample_text):
    tok1, _ = tokenizers
    texts = [sample_text[:1000] for _ in range(5)]
    ds = Dataset.from_dict({"text": texts})

    def tokenize_batch(batch):
        return {"input_ids": [tok1.tokenize(t) for t in batch["text"]]}

    mapped_ds = ds.map(tokenize_batch, batched=True)
    for item in mapped_ds:
        assert len(item["input_ids"]) > 0
    print("Dataset map test passed.")
