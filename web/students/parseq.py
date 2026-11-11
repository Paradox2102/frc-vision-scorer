# PARSeq model code vendored from https://github.com/baudm/parseq (strhub/models/parseq/
# model.py, modules.py and the permutation helpers of system.py), commit at HEAD on 2026-11-11.
#
#   Scene Text Recognition Model Hub
#   Copyright 2022 Darwin Bautista
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
#   except in compliance with the License. You may obtain a copy of the License at
#   https://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software distributed under the
#   License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Changes from upstream: no pytorch-lightning / hydra / strhub imports; the model is built with a
# digits-only vocabulary ([E] 0-9 [B] [P]) whose embeddings and head rows are sliced out of the
# 94-char pretrained checkpoint (``from_pretrained``); teacher-forced sequence log-likelihood
# (``score_strings``) for roster-constrained decoding; PLM loss as a plain function.
"""PARSeq (Bautista & Atienza, ECCV 2022) restricted to digit strings."""
from __future__ import annotations

import math
from itertools import permutations
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules import transformer
from timm.models.vision_transformer import PatchEmbed, VisionTransformer

PRETRAINED_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt"
PRETRAINED_CHARSET = ("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
IMG_H, IMG_W = 32, 128
BASE_CFG = dict(img_size=(IMG_H, IMG_W), patch_size=(4, 8), embed_dim=384, enc_num_heads=6,
                enc_mlp_ratio=4, enc_depth=12, dec_num_heads=12, dec_mlp_ratio=4, dec_depth=1,
                dropout=0.1)


class DigitTokenizer:
    """[E]=0, '0'..'9'=1..10, [B]=11, [P]=12 (same layout as upstream: EOS first, BOS/PAD last)."""
    charset = "0123456789"

    def __init__(self) -> None:
        self.itos = ("[E]",) + tuple(self.charset) + ("[B]", "[P]")
        self.eos_id, self.bos_id, self.pad_id = 0, len(self.itos) - 2, len(self.itos) - 1

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, labels: Sequence[str], device=None, length: Optional[int] = None) -> Tensor:
        """[B] digits [E] [P]..., padded to the longest label (or ``length`` + 2)."""
        L = (max(len(s) for s in labels) if length is None else length) + 2
        out = torch.full((len(labels), L), self.pad_id, dtype=torch.long)
        for i, s in enumerate(labels):
            ids = [self.bos_id] + [1 + int(c) for c in s] + [self.eos_id]
            out[i, :len(ids)] = torch.as_tensor(ids)
        return out.to(device) if device is not None else out

    def decode_ids(self, ids: Sequence[int]) -> str:
        s = []
        for i in ids:
            if i == self.eos_id or i > 10:
                break
            s.append(self.charset[i - 1])
        return "".join(s)


def _init_weights(module: nn.Module, name: str = "", exclude: Sequence[str] = ()) -> None:
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _named_apply(fn, module: nn.Module, name: str = "") -> None:
    for child_name, child in module.named_children():
        _named_apply(fn, child, f"{name}.{child_name}" if name else child_name)
    fn(module=module, name=name)


class DecoderLayer(nn.Module):
    """Pre-LN transformer decoder layer with two-stream attention (XLNet)."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = transformer._get_activation_fn("gelu")

    def forward_stream(self, tgt, tgt_norm, tgt_kv, memory, tgt_mask, tgt_key_padding_mask):
        tgt2, _ = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask, need_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt2, _ = self.cross_attn(self.norm1(tgt), memory, memory, need_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        return tgt + self.dropout3(tgt2)

    def forward(self, query, content, memory, query_mask=None, content_mask=None,
                content_key_padding_mask=None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask,
                                    content_key_padding_mask)
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)
        return query, content


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask=None, content_mask=None,
                content_key_padding_mask=None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask,
                                 content_key_padding_mask, update_content=not last)
        return self.norm(query)


class Encoder(VisionTransformer):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio):
        super().__init__(img_size, patch_size, 3, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=True, embed_layer=PatchEmbed, num_classes=0,
                         global_pool="", class_token=False)

    def forward(self, x):
        return self.forward_features(x)


class TokenEmbedding(nn.Module):
    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class PARSeq(nn.Module):
    def __init__(self, max_label_length: int = 6, img_size=(IMG_H, IMG_W), patch_size=(4, 8),
                 embed_dim=384, enc_num_heads=6, enc_mlp_ratio=4, enc_depth=12, dec_num_heads=12,
                 dec_mlp_ratio=4, dec_depth=1, dropout=0.1):
        super().__init__()
        self.tok = DigitTokenizer()
        self.max_label_length = max_label_length
        self.encoder = Encoder(img_size, patch_size, embed_dim, enc_depth, enc_num_heads, enc_mlp_ratio)
        layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))
        self.head = nn.Linear(embed_dim, len(self.tok) - 2)   # never predicts [B] / [P]
        self.text_embed = TokenEmbedding(len(self.tok), embed_dim)
        self.pos_queries = nn.Parameter(torch.empty(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        _named_apply(lambda module, name: _init_weights(module, name, exclude=["encoder"]), self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    # ------------------------------------------------------------------ pretrained weights
    @classmethod
    def from_pretrained(cls, state_dict: Optional[dict] = None, max_label_length: int = 6,
                        **kw) -> "PARSeq":
        """Base PARSeq weights with the 94-char vocab sliced down to [E] 0-9 [B] [P]."""
        if state_dict is None:
            state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_URL, map_location="cpu",
                                                            check_hash=True)
        m = cls(max_label_length=max_label_length, **kw)
        sd = dict(state_dict)
        n_chars = len(PRETRAINED_CHARSET)
        digit_rows = [1 + PRETRAINED_CHARSET.index(c) for c in DigitTokenizer.charset]
        # upstream embedding layout: [E]=0, chars 1..94, [B]=95, [P]=96; head: [E]=0, chars 1..94
        sd["text_embed.embedding.weight"] = sd["text_embed.embedding.weight"][
            [0] + digit_rows + [n_chars + 1, n_chars + 2]]
        sd["head.weight"] = sd["head.weight"][[0] + digit_rows]
        sd["head.bias"] = sd["head.bias"][[0] + digit_rows]
        sd["pos_queries"] = sd["pos_queries"][:, :max_label_length + 1]
        m.load_state_dict(sd)
        return m

    # ------------------------------------------------------------------ core
    def encode(self, img: Tensor) -> Tensor:
        return self.encoder(img)

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, tgt_query=None,
               tgt_query_mask=None):
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    @torch.no_grad()
    def greedy(self, memory: Tensor) -> tuple:
        """Greedy AR decode -> (strings, sequence log-prob of each string)."""
        bs, dev = memory.shape[0], memory.device
        steps = self.max_label_length + 1
        mask = torch.triu(torch.ones((steps, steps), dtype=torch.bool, device=dev), 1)
        tgt_in = torch.full((bs, steps), self.tok.pad_id, dtype=torch.long, device=dev)
        tgt_in[:, 0] = self.tok.bos_id
        lp_tot = torch.zeros(bs, device=dev)
        done = torch.zeros(bs, dtype=torch.bool, device=dev)
        pos = self.pos_queries[:, :steps].expand(bs, -1, -1)
        out_ids = []
        for i in range(steps):
            j = i + 1
            o = self.decode(tgt_in[:, :j], memory, mask[:j, :j], tgt_query=pos[:, i:j],
                            tgt_query_mask=mask[i:j, :j])
            lp = self.head(o)[:, 0].float().log_softmax(-1)
            if i == steps - 1:            # force [E] at the last slot
                nxt = torch.full((bs,), self.tok.eos_id, dtype=torch.long, device=dev)
            else:
                nxt = lp.argmax(-1)
            lp_tot = lp_tot + torch.where(done, torch.zeros_like(lp_tot), lp.gather(1, nxt[:, None])[:, 0])
            done = done | (nxt == self.tok.eos_id)
            out_ids.append(nxt)
            if j < steps:
                tgt_in[:, j] = nxt
            if bool(done.all()):
                break
        ids = torch.stack(out_ids, 1).tolist()
        return [self.tok.decode_ids(r) for r in ids], lp_tot

    def score_strings(self, memory: Tensor, cands: List[List[str]]) -> Tensor:
        """Teacher-forced log P(string, [E] | image) under the forward (left-to-right) order.
        ``memory`` [N, T, D]; ``cands`` N lists of K digit strings -> [N, K] log-likelihoods."""
        N, K = len(cands), len(cands[0])
        flat = [s for c in cands for s in c]
        tgt = self.tok.encode(flat, memory.device, length=self.max_label_length)  # [N*K, L+2]
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        L = tgt_in.shape[1]
        mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=memory.device), 1)
        mem = memory.repeat_interleave(K, dim=0)
        out = self.decode(tgt_in, mem, mask, None, tgt_query_mask=mask)
        lp = self.head(out).float().log_softmax(-1)
        valid = tgt_out != self.tok.pad_id
        tok_lp = lp.gather(2, tgt_out.clamp(max=lp.shape[-1] - 1)[..., None])[..., 0]
        return (tok_lp * valid).sum(1).view(N, K)


# ---------------------------------------------------------------------- permutation LM loss
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
