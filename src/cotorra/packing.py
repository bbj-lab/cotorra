#!/usr/bin/env python3

"""
admission-level metadata for packed training windows;
each subject timeline (one hospital admission in the CLIF schema) occupies a
contiguous run of tokens inside a packed window, identified by the
`admission_ids` emitted in `cotorra.util.batched_iter`
"""

import torch as t


def admission_relative_position_ids(admission_ids: t.Tensor) -> t.Tensor:
    """
    position indices that restart at 0 at the first token of every admission;
    e.g. admission_ids [7, 7, 7, 8, 8] -> [0, 1, 2, 0, 1]
    """
    batch_size, seq_len = admission_ids.shape
    seq_idx = (
        t.arange(seq_len, device=admission_ids.device)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
    )
    is_start = t.ones_like(admission_ids, dtype=t.bool)
    is_start[:, 1:] = admission_ids[:, 1:] != admission_ids[:, :-1]
    admission_start_idx = t.where(is_start, seq_idx, 0).cummax(dim=-1).values
    return seq_idx - admission_start_idx


def mask_cross_admission_labels(
    input_ids: t.Tensor, admission_ids: t.Tensor
) -> t.Tensor:
    """
    next-token labels with -100 wherever the prediction target belongs to a
    different admission than its source position; under the HF shift
    (logits[:, :-1] vs labels[:, 1:]) this stops CE from training the last
    token of one admission to predict the first token of the next
    """
    labels = input_ids.clone()
    crosses = admission_ids[:, 1:] != admission_ids[:, :-1]
    labels[:, 1:] = labels[:, 1:].masked_fill(crosses, -100)
    return labels


def nextlat_transition_mask(admission_ids: t.Tensor) -> t.Tensor:
    """
    boolean mask of shape (batch, seq_len - 1); True at index t iff the
    horizon-1 transition (h_t, x_{t+1}) -> h_{t+1} stays inside one admission
    """
    return admission_ids[:, 1:] == admission_ids[:, :-1]
