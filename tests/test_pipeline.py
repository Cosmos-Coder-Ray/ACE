"""Real-data, artifact and cheap-weight-operation integration tests."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ace.data.collator import CausalCollator
from ace.data.dataset import PackedTokenDataset
from ace.data.preprocessing import prepare_corpus
from ace.inference.bundle import load_bundle, save_bundle
from ace.inference.generate import generate_ids
from ace.merging import merge_bundles, upcycle_bundle
from ace.model.backbone import AceModel
from ace.model.config import AceConfig
from ace.tokenizer.ace_tokenizer import AceTokenizer
from scripts.evaluate import evaluate_model


@pytest.fixture
def prepared(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "tiny_code.jsonl"
    config = AceConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "ace_smoke.yaml")
    prepare_corpus(
        [source, source],
        tmp_path / "data",
        config,
        validation_fraction=0.2,
        source_license="project-authored test data",
    )
    return tmp_path / "data"


def test_unseen_bytes_are_lossless(tmp_path):
    corpus = tmp_path / "simple.txt"
    corpus.write_text("aaa aaa\n", encoding="utf-8")
    tokenizer = AceTokenizer.train([corpus], vocab_size=300)
    for text in ("print(123)", "हिन्दी", "你好", "😀", "e\u0301", "x\r\n\ty"):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_document_split_and_exact_dedup(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    hashes = [
        {
            json.loads(line)["sha256"]
            for line in (prepared / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        }
        for split in ("train", "validation")
    ]
    assert hashes[0].isdisjoint(hashes[1])
    assert manifest["exact_duplicates_removed"] == 30
    assert sum(manifest["documents"].values()) == 30
    assert manifest["vocab_size"] == AceConfig.from_yaml(prepared / "config.yaml").vocab_size


def test_packing_preserves_every_target_and_padding_is_masked(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(11, dtype="<u4").tofile(path)
    dataset = PackedTokenDataset(path, seq_len=4, vocab_size=12)
    examples = [dataset[index] for index in range(len(dataset))]
    targets = torch.cat([x["input_ids"][1:] for x in examples])
    assert targets.tolist() == list(range(1, 11))
    batch = CausalCollator(0)(examples)
    assert (batch["labels"][~batch["attention_mask"][:, 0, 0]] == -100).all()


def test_bundle_generation_merge_and_upcycling(prepared, tmp_path):
    tokenizer = AceTokenizer.load(prepared / "tokenizer")
    config = replace(AceConfig.from_yaml(prepared / "config.yaml"), moe_layers=[])
    torch.manual_seed(7)
    model = AceModel(config).eval()
    left = tmp_path / "left"
    save_bundle(model, tokenizer, left, lineage_id="common-base")
    restored, _, _ = load_bundle(left)
    ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode("def add")])
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, restored(ids).logits)
    generated = generate_ids(restored, ids, max_new_tokens=3)
    assert generated.shape[1] == ids.shape[1] + 3
    assert generate_ids(restored, ids, max_new_tokens=3).equal(generated)

    with torch.no_grad():
        model.layers[0].ffn.gate_proj.weight.add_(0.1)
    right = tmp_path / "right"
    save_bundle(model, tokenizer, right, lineage_id="common-base")
    merged_path = tmp_path / "merged"
    merge_bundles(left, right, merged_path, alpha=0.25)
    merged, _, _ = load_bundle(merged_path)
    torch.testing.assert_close(
        merged.layers[0].ffn.gate_proj.weight, restored.layers[0].ffn.gate_proj.weight + 0.025
    )
    expanded_path = tmp_path / "expanded"
    expanded_meta = upcycle_bundle(left, expanded_path, [1], num_experts=4, top_k=2)
    expanded, _, _ = load_bundle(expanded_path)
    with torch.no_grad():
        torch.testing.assert_close(expanded(ids).logits, restored(ids).logits, atol=1e-6, rtol=1e-5)
    assert expanded_meta["parameters"] > restored.num_parameters

    save_bundle(model, tokenizer, tmp_path / "unrelated", lineage_id="different-base")
    with pytest.raises(ValueError, match="lineage_id"):
        merge_bundles(left, tmp_path / "unrelated", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_evaluation_token_count(prepared):
    config = AceConfig.from_yaml(prepared / "config.yaml")
    tokenizer = AceTokenizer.load(prepared / "tokenizer")
    data = PackedTokenDataset(prepared / "validation.bin", 32, config.vocab_size)
    loader = DataLoader(data, batch_size=2, collate_fn=CausalCollator(tokenizer.pad_token_id))
    result = evaluate_model(AceModel(config), loader)
    assert result["tokens"] == len(data.tokens) - 1
    assert result["cross_entropy"] > 0
    assert np.isfinite(result["perplexity"])
