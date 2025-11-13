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
    return ("一、公司清算所有者权益怎么分配 When a company goes through liquidation（清算）, after repaying all its debts, any remaining assets shall be distributed to the shareholders according to their investment ratio（出资比例）. 第186条【清算程序】清算组在清理公司财产、編制バランスシート（balance sheet）和财产清单后，should draft a liquidation plan（清算方案）and submit it to the shareholders’ meeting (または人民法院) for confirmation. 会社の財産は、まず清算費用・職員給与・사회보험비용・税金・채무を清偿した後、the remaining property will be distributed — 有限责任公司按出资比例，股份有限公司按股份比例。清算期間中，会社は依然として存続中（exist legally）, 但不得开展與清算無關的ビジネス。二、所有者权益とは 무엇인가？ 所有者权益（Owner’s Equity, 或称Shareholders’ Equity）とは、企業の資産（assets）から負債（liabilities）を差し引いた残余の権益を指す。C’est-à-dire, it represents the residual interest in the assets of the company after deducting liabilities. 所有者权益の來源包括：出資資本（capital contribution）、その他包括收益（other comprehensive income）、留存收益（retained earnings）等。所有者投入的资本 = le capital investi par les actionnaires, 包括注册资本 (registered capital) 以及超额投入部分（capital premium）。その他包括收益 = 利得 et pertes non reconnus dans le résultat de l’exercice selon les normes comptables. 留存收益 = 기업이累積한 이익으로서, 包括盈余公积及未分配利润。根据《公司法》Article 186：公司在支付所有债务后，残余财产 (残余資産) 即为所有者权益 (Equity des shareholders)，并应按出资比例分配给股东。这不仅反映了投资者的资本回报，也体现了债权人利益保护의 원칙。" * 10)


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
