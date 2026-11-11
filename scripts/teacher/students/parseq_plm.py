# PARSeq permutation-LM training loss vendored from https://github.com/baudm/parseq
# (strhub/models/parseq/system.py), commit at HEAD on 2026-11-11. The model itself is in
# web/students/parseq.py (runtime); this file is training-only.
#
#   Scene Text Recognition Model Hub
#   Copyright 2022 Darwin Bautista
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
#   except in compliance with the License. You may obtain a copy of the License at
#   https://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software distributed under the
#   License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Changes from upstream: no pytorch-lightning / hydra / strhub imports; PLM loss as a plain function.
"""PARSeq permutation language modelling loss (training only; S1 digits.py)."""
from __future__ import annotations

import math
from itertools import permutations
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from web.students.parseq import PARSeq


def gen_tgt_perms(tgt: Tensor, perm_num: int = 6, perm_forward: bool = True,
                  perm_mirrored: bool = True, rng: Optional[np.random.Generator] = None) -> Tensor:
    """Upstream ``PARSeq.gen_tgt_perms`` (shared permutations for the whole batch)."""
    rng = rng or np.random.default_rng()
    dev = tgt.device
    max_num_chars = tgt.shape[1] - 2
    if max_num_chars == 1:
        return torch.arange(3, device=dev).unsqueeze(0)
    max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
    perms = [torch.arange(max_num_chars, device=dev)] if perm_forward else []
    max_perms = math.factorial(max_num_chars)
    if perm_mirrored:
        max_perms //= 2
    num_gen_perms = min(max_gen_perms, max_perms)
    if max_num_chars < 5:
        if max_num_chars == 4 and perm_mirrored:
            selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
        else:
            selector = list(range(max_perms))
        perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)),
                                    device=dev)[selector]
        if perm_forward:
            perm_pool = perm_pool[1:]
        perms = torch.stack(perms)
        if len(perm_pool) and num_gen_perms - len(perms) > 0:
            i = rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
            perms = torch.cat([perms, perm_pool[i]])
    else:
        perms.extend([torch.randperm(max_num_chars, device=dev) for _ in range(num_gen_perms - len(perms))])
        perms = torch.stack(perms)
    if perm_mirrored:
        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
    bos_idx = perms.new_zeros((len(perms), 1))
    eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
    perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
    if len(perms) > 1:
        perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=dev)
    return perms


def generate_attn_masks(perm: Tensor):
    sz = perm.shape[0]
    mask = torch.zeros((sz, sz), dtype=torch.bool, device=perm.device)
    for i in range(sz):
        mask[perm[i], perm[i + 1:]] = True
    content_mask = mask[:-1, :-1].clone()
    mask[torch.eye(sz, dtype=torch.bool, device=perm.device)] = True
    return content_mask, mask[1:, :-1]


def plm_loss(model: PARSeq, images: Tensor, labels: Sequence[str],
             rng: Optional[np.random.Generator] = None, perm_num: int = 6) -> Tensor:
    """Upstream ``PARSeq.training_step`` loss (permutation language modelling)."""
    tok = model.tok
    tgt = tok.encode(labels, images.device)
    memory = model.encode(images)
    tgt_perms = gen_tgt_perms(tgt, perm_num=perm_num, rng=rng)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    pad_mask = (tgt_in == tok.pad_id) | (tgt_in == tok.eos_id)
    loss, numel = 0.0, 0
    n = (tgt_out != tok.pad_id).sum().item()
    for i, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = generate_attn_masks(perm)
        out = model.decode(tgt_in, memory, tgt_mask, pad_mask, tgt_query_mask=query_mask)
        logits = model.head(out).flatten(end_dim=1).float()
        loss = loss + n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=tok.pad_id)
        numel += n
        if i == 1:
            tgt_out = torch.where(tgt_out == tok.eos_id, tok.pad_id, tgt_out)
            n = (tgt_out != tok.pad_id).sum().item()
    return loss / numel
