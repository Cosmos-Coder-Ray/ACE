"""Local greedy or sampled generation from a portable ACE bundle."""

import argparse
import math
from pathlib import Path

import torch

from ace.inference.bundle import load_bundle
from ace.model.backbone import AceModel


@torch.inference_mode()
def generate_ids(
    model: AceModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 64,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 42,
) -> torch.Tensor:
    """Return prompt plus completion. Hybrid models recompute their full prefix."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Generation currently supports one nonempty prompt")
    if max_new_tokens < 0 or temperature < 0 or not math.isfinite(temperature) or top_k < 0:
        raise ValueError("Invalid generation settings")
    if input_ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("Prompt plus generation exceeds configured context; shorten the request")
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    training = model.training
    model.eval()
    sequence = input_ids.clone()
    use_cache = not model.config.mamba_layers
    cache = None
    try:
        for _ in range(max_new_tokens):
            tokens = sequence[:, -1:] if cache is not None else sequence
            result = model(tokens, past_key_values=cache, use_cache=use_cache)
            cache = result.past_key_values if use_cache else None
            logits = result.logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                next_id = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            sequence = torch.cat((sequence, next_id), dim=1)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
    finally:
        model.train(training)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, tokenizer, _ = load_bundle(args.model, args.device)
    ids = [tokenizer.bos_token_id] + tokenizer.encode(args.prompt)
    output = generate_ids(
        model,
        torch.tensor([ids], device=args.device),
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
