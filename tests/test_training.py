"""Exercise real optimizer updates and exact CPU resumption across sample epochs."""

from dataclasses import replace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from ace.data.collator import CausalCollator
from ace.data.samplers import ResumableRandomSampler
from ace.model.backbone import AceModel
from ace.training.trainer import AceTrainer, TrainConfig


def test_resume_matches_uninterrupted_training(micro_config, tmp_path):
    torch.set_num_threads(2)
    model_config = replace(micro_config, n_layers=2, moe_layers=[], dropout=0.1)
    torch.manual_seed(9)
    initial = AceModel(model_config).state_dict()
    examples = [{"input_ids": torch.tensor([2, i, 4, 5, 6, 7])} for i in range(3, 8)]

    def make_loader():
        return DataLoader(
            examples,
            batch_size=2,
            sampler=ResumableRandomSampler(len(examples), 11),
            collate_fn=CausalCollator(0),
            generator=torch.Generator().manual_seed(11),
        )

    def make_model():
        model = AceModel(model_config)
        model.load_state_dict(initial)
        return model

    config = TrainConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=4,
        warmup_steps=1,
        checkpoint_every=2,
        log_every=1,
        bf16=False,
        checkpoint_dir=str(tmp_path / "full"),
    )
    full = make_model()
    trainer = AceTrainer(full, model_config, config)
    torch.manual_seed(25)
    trainer.train(make_loader())

    interrupted_config = replace(config, checkpoint_dir=str(tmp_path / "partial"), max_seconds=2.5)
    partial = make_model()
    trainer = AceTrainer(partial, model_config, interrupted_config)
    torch.manual_seed(25)
    # The time budget stops after the second complete optimizer step.
    with patch("ace.training.trainer.time.monotonic", side_effect=[0, 1, 2, 3]):
        assert trainer.train(make_loader())["final_step"] == 2
    resumed = make_model()
    resume_config = replace(
        interrupted_config, max_seconds=None, resume_from_checkpoint=str(tmp_path / "partial")
    )
    trainer = AceTrainer(resumed, model_config, resume_config)
    assert trainer.train(make_loader())["final_step"] == 4
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
    saved_loss = trainer.checkpoint_mgr.load_latest()["loss"]
    completed = AceTrainer(make_model(), model_config, resume_config)
    assert completed.train(make_loader())["final_loss"] == saved_loss


def test_training_learns_a_repeated_pattern(micro_config, tmp_path):
    torch.set_num_threads(2)
    config = replace(micro_config, n_layers=2, moe_layers=[])
    torch.manual_seed(1)
    model = AceModel(config)
    batch = {"input_ids": torch.tensor([[2, 3, 4, 5, 2, 3, 4, 5]])}
    trainer = AceTrainer(
        model,
        config,
        TrainConfig(
            max_steps=20,
            warmup_steps=1,
            lr=0.01,
            gradient_accumulation_steps=1,
            bf16=False,
            checkpoint_dir=str(tmp_path),
        ),
    )
    from ace.training.loss import cross_entropy_loss

    with torch.no_grad():
        before = cross_entropy_loss(model(batch["input_ids"]).logits, batch["input_ids"]).item()
    for _ in range(20):
        trainer.optimizer.zero_grad()
        trainer._train_step(batch)
        trainer.optimizer.step()
        trainer.scheduler.step()
    with torch.no_grad():
        after = cross_entropy_loss(model(batch["input_ids"]).logits, batch["input_ids"]).item()
    assert after < before * 0.5
