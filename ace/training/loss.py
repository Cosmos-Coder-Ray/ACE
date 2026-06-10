"""Loss computation for ACE pretraining.

This module provides the loss functions used during ACE pretraining,
combining autoregressive language-model cross-entropy loss with MoE
auxiliary load-balancing loss.  The autoregressive loss shifts logits
and labels by one position so that position *t* predicts token *t+1*.
The auxiliary loss (typically produced by the MoE router) is scaled by
a small coefficient and added to the LM loss to encourage balanced
expert utilisation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Compute autoregressive (causal-LM) cross-entropy loss.

    Logits and labels are **shifted** internally so that position *t*
    in the logits predicts the token at position *t+1* in the labels.

    Parameters
    ----------
    logits : Tensor
        Raw (unscaled) predictions of shape ``(B, L, V)``.
    labels : Tensor
        Ground-truth token ids of shape ``(B, L)``.
    ignore_index : int, optional
        Label value that should be ignored when computing the loss
        (default ``-100``).

    Returns
    -------
    Tensor
        Scalar mean cross-entropy loss.

    Raises
    ------
    ValueError
        If ``logits`` is not 3-D or ``labels`` is not 2-D, or if their
        batch / sequence dimensions do not match.
    """
    # -- Validate shapes ---------------------------------------------------
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3-D (B, L, V), got ndim={logits.ndim}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D (B, L), got ndim={labels.ndim}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Batch size mismatch: logits batch={logits.shape[0]}, "
            f"labels batch={labels.shape[0]}"
        )
    if logits.shape[1] != labels.shape[1]:
        raise ValueError(
            f"Sequence length mismatch: logits seq={logits.shape[1]}, "
            f"labels seq={labels.shape[1]}"
        )

    # -- Shift for autoregressive prediction --------------------------------
    shift_logits = logits[:, :-1, :]  # (B, L-1, V)
    shift_labels = labels[:, 1:]  # (B, L-1)

    batch_size = shift_logits.shape[0]  # B
    seq_len_minus_one = shift_logits.shape[1]  # L-1
    vocab_size = shift_logits.shape[2]  # V

    # Flatten to 2-D / 1-D for F.cross_entropy
    shift_logits = shift_logits.reshape(batch_size * seq_len_minus_one, vocab_size)  # (B*(L-1), V)
    shift_labels = shift_labels.reshape(batch_size * seq_len_minus_one)  # (B*(L-1),)

    # Cast to float32 for numerical stability
    shift_logits = shift_logits.float()  # (B*(L-1), V)

    loss: Tensor = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
    )  # scalar
    return loss


def compute_total_loss(
    logits: Tensor,
    labels: Tensor,
    aux_loss: Tensor,
    ignore_index: int = -100,
    aux_loss_coeff: float = 0.01,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the combined pretraining loss.

    The total loss is the sum of the autoregressive LM loss and the
    (scaled) MoE auxiliary load-balancing loss:

        ``total = lm_loss + aux_loss_coeff * aux_loss``

    Parameters
    ----------
    logits : Tensor
        Raw predictions, shape ``(B, L, V)``.
    labels : Tensor
        Ground-truth token ids, shape ``(B, L)``.
    aux_loss : Tensor
        Scalar auxiliary loss produced by the MoE routing layer.
    ignore_index : int, optional
        Label value to ignore (default ``-100``).
    aux_loss_coeff : float, optional
        Coefficient for the auxiliary loss (default ``0.01``).

    Returns
    -------
    tuple[Tensor, dict[str, float]]
        ``(total_loss, metrics)`` where *metrics* is a dictionary with
        keys ``"lm_loss"``, ``"aux_loss"``, and ``"total_loss"`` whose
        values are plain Python floats suitable for logging.

    Raises
    ------
    ValueError
        If ``aux_loss_coeff`` is negative.
    """
    if aux_loss_coeff < 0:
        raise ValueError(f"aux_loss_coeff must be >= 0, got {aux_loss_coeff}")

    lm_loss: Tensor = cross_entropy_loss(logits, labels, ignore_index=ignore_index)  # scalar

    total_loss: Tensor = lm_loss + aux_loss_coeff * aux_loss  # scalar

    metrics: dict[str, float] = {
        "lm_loss": lm_loss.item(),
        "aux_loss": aux_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    torch.manual_seed(42)

    B, L, V = 2, 16, 256

    # ---- Test 1: cross_entropy_loss produces scalar, non-neg, non-NaN ----
    logits = torch.randn(B, L, V)  # (B, L, V)
    labels = torch.randint(0, V, (B, L))  # (B, L)

    loss = cross_entropy_loss(logits, labels)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() >= 0.0, "Loss should be non-negative"
    assert not math.isnan(loss.item()), "Loss should not be NaN"
    print("[PASS] cross_entropy_loss produces scalar, non-negative, non-NaN loss")

    # ---- Test 2: ignore_index changes loss value -------------------------
    loss_before = cross_entropy_loss(logits, labels).item()
    labels_masked = labels.clone()
    labels_masked[:, :4] = -100  # mask first 4 positions
    loss_after = cross_entropy_loss(logits, labels_masked).item()
    assert loss_before != loss_after, "Loss should change when some labels are masked"
    print("[PASS] ignore_index correctly changes loss value")

    # ---- Test 3: compute_total_loss returns correct dict keys & total ----
    aux = torch.tensor(0.5, requires_grad=True)
    coeff = 0.01
    total, metrics = compute_total_loss(logits, labels, aux, aux_loss_coeff=coeff)
    assert set(metrics.keys()) == {
        "lm_loss",
        "aux_loss",
        "total_loss",
    }, f"Unexpected metrics keys: {metrics.keys()}"
    expected_total = metrics["lm_loss"] + coeff * metrics["aux_loss"]
    assert math.isclose(
        metrics["total_loss"], expected_total, rel_tol=1e-5
    ), f"total_loss mismatch: {metrics['total_loss']} vs {expected_total}"
    print("[PASS] compute_total_loss returns correct dict keys and total = lm + coeff*aux")

    # ---- Test 4: gradient flows through both losses ----------------------
    logits_g = torch.randn(B, L, V, requires_grad=True)  # (B, L, V)
    labels_g = torch.randint(0, V, (B, L))  # (B, L)
    aux_g = torch.tensor(1.0, requires_grad=True)

    total_g, _ = compute_total_loss(logits_g, labels_g, aux_g, aux_loss_coeff=0.01)
    total_g.backward()  # type: ignore[no-untyped-call]

    assert logits_g.grad is not None, "Gradient should flow to logits"
    assert aux_g.grad is not None, "Gradient should flow to aux_loss"
    assert logits_g.grad.shape == logits_g.shape, "Gradient shape mismatch for logits"
    print("[PASS] Gradient flows through both LM and auxiliary losses")

    # ---- Test 5: validation errors (wrong shapes) ------------------------
    try:
        cross_entropy_loss(torch.randn(B, L), labels)
        raise AssertionError("Should have raised ValueError for 2-D logits")
    except ValueError:
        pass
    print("[PASS] ValueError raised for logits with wrong ndim")

    try:
        cross_entropy_loss(torch.randn(B, L, V), torch.randint(0, V, (B,)))
        raise AssertionError("Should have raised ValueError for 1-D labels")
    except ValueError:
        pass
    print("[PASS] ValueError raised for labels with wrong ndim")

    try:
        cross_entropy_loss(torch.randn(B, L, V), torch.randint(0, V, (B + 1, L)))
        raise AssertionError("Should have raised ValueError for batch mismatch")
    except ValueError:
        pass
    print("[PASS] ValueError raised for batch size mismatch")

    try:
        cross_entropy_loss(torch.randn(B, L, V), torch.randint(0, V, (B, L + 1)))
        raise AssertionError("Should have raised ValueError for seq len mismatch")
    except ValueError:
        pass
    print("[PASS] ValueError raised for sequence length mismatch")

    # ---- Test 6: negative aux_loss_coeff ---------------------------------
    try:
        compute_total_loss(logits, labels, torch.tensor(0.5), aux_loss_coeff=-0.1)
        raise AssertionError("Should have raised ValueError for negative coeff")
    except ValueError:
        pass
    print("[PASS] ValueError raised for negative aux_loss_coeff")

    print("\nAll smoke tests passed.")
