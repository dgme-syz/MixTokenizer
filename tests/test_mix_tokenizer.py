import os
import time
import pytest
import random

from transformers import AutoTokenizer
from datasets import Dataset
import pickle
import tempfile


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
    return ("你好akhdkas ajda !ajca你好cajdasljcjlaccaslcl!javcj  ajlda &&&&&jacl" * 1000)


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
    decoded = tok1.decode(tokens1, map_back=False)
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
        return {"tonize": [tok1.tokenize(t) for t in batch["text"]]}

    mapped_ds = ds.map(tokenize_batch, batched=True, load_from_cache_file=False, cache_file_name=None,)
    for item in mapped_ds:
        assert len(item["tonize"]) > 0
    print("Dataset map test passed.")
