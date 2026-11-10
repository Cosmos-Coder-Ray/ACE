import json
from pathlib import Path

import pytest

from ace.data.supervised import SupervisedDataset, encode_supervised_record
from ace.tokenizer.ace_tokenizer import AceTokenizer
from ace.tokenizer.special_tokens import FIM_PREFIX, SPECIAL_TOKENS_DICT


@pytest.fixture
def tokenizer(tmp_path):
    source = tmp_path / "code.txt"
    source.write_text(
        "def add(a, b):\n    return a + b\n"
        "def multiply(a, b):\n    return a * b\n" * 20,
        encoding="utf-8",
    )
    return AceTokenizer.train([source], vocab_size=512, min_frequency=1)


def test_instruction_labels_only_assistant_response_and_end(tokenizer):
    record = {
        "kind": "instruction",
        "system": "You edit code.",
        "user": "Fix add.",
        "assistant": "return a + b",
    }
    result = encode_supervised_record(record, tokenizer, 128)
    assistant_id = tokenizer.token_to_id(SPECIAL_TOKENS_DICT["ASSISTANT"])
    assistant_position = result["input_ids"].tolist().index(assistant_id)

    assert (result["labels"][: assistant_position + 1] == -100).all()
    assert (result["labels"][assistant_position + 1 :] != -100).all()
    assert result["labels"][-1].item() == tokenizer.eos_token_id


def test_fim_labels_only_middle_and_end(tokenizer):
    record = {
        "kind": "fim",
        "prefix": "def add(a, b):\n",
        "suffix": "\nprint(add(2, 3))\n",
        "middle": "    return a + b",
    }
    result = encode_supervised_record(record, tokenizer, 128)
    middle_id = tokenizer.token_to_id(SPECIAL_TOKENS_DICT["FIM_MIDDLE"])
    middle_position = result["input_ids"].tolist().index(middle_id)

    assert (result["labels"][: middle_position + 1] == -100).all()
    assert (result["labels"][middle_position + 1 :] != -100).all()
    assert result["labels"][-1].item() == tokenizer.eos_token_id


def test_legacy_tokenizer_keeps_its_original_vocabulary():
    legacy_path = Path(__file__).resolve().parents[1] / "data" / "tokenizer"
    legacy = AceTokenizer.load(legacy_path)
    record = {"kind": "fim", "prefix": "a", "suffix": "c", "middle": "b"}
    assert legacy.format_version == 1
    assert legacy.token_to_id(FIM_PREFIX) is None
    with pytest.raises(ValueError, match="FIM_PREFIX token"):
        encode_supervised_record(record, legacy, 32)


def test_dataset_reads_instruction_records(tmp_path, tokenizer):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "instruction",
                "system": "You edit code.",
                "user": "Fix add.",
                "assistant": "return a + b",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SupervisedDataset(path, tokenizer, 128)

    assert len(dataset) == 1
    assert dataset[0]["input_ids"].shape == dataset[0]["labels"].shape


def test_instruction_record_requires_target(tokenizer):
    with pytest.raises(ValueError, match="string system, user, and assistant"):
        encode_supervised_record({"kind": "instruction", "user": "Fix"}, tokenizer, 32)
