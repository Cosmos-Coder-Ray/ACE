"""Load either an ACE bundle or a separate Transformers adapter for inference."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ace.inference.bundle import load_bundle
from ace.inference.generate import generate_ids
from ace.data.preprocessing import file_sha256


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


class ModelRuntime:
    def __init__(self, model_path: Path, *, threads: int = 2) -> None:
        torch.set_num_threads(threads)
        self.model_path = model_path.resolve()
        self.device = torch.device("cpu")
        self.track: str
        self.model: Any
        self.tokenizer: Any
        self.context_length: int | None = None

        if (self.model_path / "config.yaml").is_file() and (
            self.model_path / "manifest.json"
        ).is_file():
            self.model, self.tokenizer, metadata = load_bundle(self.model_path, "cpu")
            self.track = "native-ace"
            self.context_length = int(metadata["config"]["max_seq_len"])
            self.model_revision = metadata.get("lineage_id", "unknown")
            self.tokenizer_revision = metadata["tokenizer_sha256"]
        else:
            self._load_adapter()

    def _load_adapter(self) -> None:
        manifest_path = self.model_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No ACE bundle or adapter manifest at {self.model_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or not manifest.get("model_revision"):
            raise ValueError("Unsupported or unpinned adapter manifest")
        if file_sha256(self.model_path / "tokenizer" / "tokenizer.json") != manifest["tokenizer_sha256"]:
            raise ValueError("Adapter tokenizer fingerprint mismatch")
        if _tree_sha256(self.model_path / "adapter") != manifest["adapter_sha256"]:
            raise ValueError("Adapter weight fingerprint mismatch")
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Pretrained inference requires pip install -e '.[finetune]'"
            ) from exc

        base_id = manifest["model_id"]
        revision = manifest["model_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path / "tokenizer", local_files_only=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            revision=revision,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base, self.model_path / "adapter")
        self.model.to(self.device).eval()
        self.track = "pretrained-lora"
        self.model_revision = revision
        self.tokenizer_revision = manifest["tokenizer_sha256"]
        self.context_length = int(getattr(self.model.config, "max_position_embeddings", 4096))

    def generate(self, prompt: str, max_new_tokens: int = 128) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024")
        if self.context_length and max_new_tokens >= self.context_length:
            raise ValueError("max_new_tokens must be shorter than the model context")

        started = time.perf_counter()
        if self.track == "native-ace":
            ids = [self.tokenizer.bos_token_id, *self.tokenizer.encode(prompt)]
            if self.context_length and len(ids) + max_new_tokens > self.context_length:
                ids = ids[-(self.context_length - max_new_tokens) :]
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            output = generate_ids(
                self.model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated_ids = output[0, input_ids.shape[1] :].tolist()
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            token_count = len(generated_ids)
        else:
            if getattr(self.tokenizer, "chat_template", None):
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            max_input = max(1, self.context_length - max_new_tokens) if self.context_length else 4096
            if inputs["input_ids"].shape[1] > max_input:
                inputs = {key: value[:, -max_input:] for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = output[0, inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            token_count = int(generated_ids.numel())

        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "text": text,
            "generated_tokens": token_count,
            "seconds": elapsed,
            "tokens_per_second": token_count / elapsed,
            "track": self.track,
        }

    @torch.inference_mode()
    def score(self, prompt: str, reference: str) -> tuple[float, int]:
        """Return token-weighted cross entropy for a held-out continuation."""
        if not prompt or not reference:
            raise ValueError("Loss scoring requires a prompt and reference continuation")
        if self.track == "native-ace":
            prompt_ids = self.tokenizer.encode(prompt)
            target_ids = self.tokenizer.encode(reference)
            ids = [self.tokenizer.bos_token_id, *prompt_ids, *target_ids, self.tokenizer.eos_token_id]
            prompt_length = 1 + len(prompt_ids)
            model = self.model
        else:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            target_ids = self.tokenizer.encode(reference, add_special_tokens=False)
            eos = self.tokenizer.eos_token_id
            ids = [*prompt_ids, *target_ids, *([int(eos)] if eos is not None else [])]
            prompt_length = len(prompt_ids)
            model = self.model
        if not target_ids or prompt_length == 0:
            raise ValueError("Loss scoring produced an empty target")
        if self.context_length and len(ids) > self.context_length:
            removed = len(ids) - self.context_length
            ids = ids[-self.context_length :]
            prompt_length -= removed
            prompt_length = max(1, prompt_length)
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = model(input_ids=tensor).logits[:, :-1].float()
        labels = tensor[:, 1:]
        target_start = max(0, prompt_length - 1)
        logits = logits[:, target_start:]
        labels = labels[:, target_start:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        count = int(labels.numel())
        return float(loss) / count, count
