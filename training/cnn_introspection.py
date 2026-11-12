#!/usr/bin/env python3
"""
Multi-Scale 1D Convolutional Neural Network (TextCNN) +
Recursive Action-Reaction Introspection Model (Pure NumPy Training -> JSON Export)
---------------------------------------------------------------------------------
Architecture:
  1. TextCNN Dual-Stream Backbone:
     - Stream A: Sequence Embedding Lookup E[tokens[:36], D=32] ->
       3 Parallel 1D Convolution Banks (kernel widths 2, 3, 4; 24 filters each) ->
       ReLU -> Global Max-Over-Time + Masked Mean-Over-Time Pooling (144 dims)
     - Stream B: L2-normalized Sublinear Multilingual TF-IDF Projection (48 dims)
     - Fusion Layer: Concat(192) -> Dense(96) + Tanh -> Softmax Domain Classifier (C classes)
     - Trained via Mini-Batch AdamW + Cosine Annealing + Label Smoothing

  2. Recursive Action-Reaction Introspection Unit (3-Step BPTT):
     - Maintains latent cognitive state z^(s) initialized from h^(0)
     - Takes candidate action one-hot a^(s) in A={conversational, math, local_kb, browse_web, compare, synthesize_deep}
       plus 12-dim empirical retrieval Reaction vector r^(s) from the knowledge/browser environment
     - Updates state via Gated Introspection Cell:
         g = sigmoid(W_g [z; r] + b_g)
         u = tanh(W_u [z; a; r] + b_u)
         z^(s+1) = g * z + (1 - g) * u
     - Outputs:
         * Next Optimal Action distribution pi^(s+1)
         * Epistemic Sufficiency Critic score q_hat^(s+1) in (0, 1)
         * Domain Logit Correction delta_l^(s+1)
"""

import re
import unicodedata
from typing import Dict, List, Tuple

import numpy as np

ACTIONS = [
    "answer_conversational",
    "evaluate_math",
    "retrieve_local_kb",
    "browse_web_multi",
    "compare_entities",
    "synthesize_deep",
]
NUM_ACTIONS = len(ACTIONS)
REACTION_DIM = 12
CNN_MAX_LEN = 36
CNN_EMB_DIM = 32
CNN_KERNELS = (2, 3, 4)
CNN_FILTERS = 24
TFIDF_PROJ_DIM = 48
HIDDEN_DIM = 96


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text).lower())
    chars = []
    for c in text:
        cat = unicodedata.category(c)
        if cat[0] in ("L", "M", "N") or c.isspace():
            chars.append(c)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3040 <= cp <= 0x309F
        or 0x30A0 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


def extract_cnn_sequence_tokens(raw_text: str) -> List[str]:
    """
    Produces ordered sequence tokens for 1D convolution kernels (matching JS cleanly):
    - Alphabetic/abugida words + `<word>` / prefix-3 subword morphemes for ASR typo resilience
    - Individual CJK characters
    """
    norm = normalize_text(raw_text)
    if not norm:
        return []
    seq: List[str] = []
    for word in norm.split():
        if any(is_cjk_char(c) for c in word):
            for ch in word:
                if is_cjk_char(ch):
                    seq.append(ch)
                elif len(seq) == 0 or seq[-1] != ch:
                    seq.append(ch)
        else:
            seq.append(word)
            if len(word) >= 6:
                seq.append(f"#{word[:4]}")
                seq.append(f"#{word[-4:]}")
        if len(seq) >= CNN_MAX_LEN:
            break
    return seq[:CNN_MAX_LEN]


def build_cnn_vocab(texts: List[str], max_vocab: int = 3000) -> Dict[str, int]:
    from collections import Counter

    counts = Counter()
    for t in texts:
        for tok in extract_cnn_sequence_tokens(t):
            counts[tok] += 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in counts.most_common(max_vocab - 2):
        vocab[tok] = len(vocab)
    return vocab


def encode_sequences(texts: List[str], cnn_vocab: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(texts)
    ids = np.zeros((n, CNN_MAX_LEN), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    unk = cnn_vocab.get("<UNK>", 1)
    for i, t in enumerate(texts):
        toks = extract_cnn_sequence_tokens(t)
        lengths[i] = max(1, len(toks))
        for j, tok in enumerate(toks[:CNN_MAX_LEN]):
            ids[i, j] = cnn_vocab.get(tok, unk)
    return ids, lengths


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


class AdamWOptimizer:
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 3e-3, weight_decay: float = 1e-4):
        self.lr = lr
        self.wd = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        # Global gradient clipping
        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        clip_coef = 3.0 / max(3.0, np.sqrt(total_sq) + 1e-12)
        for k, g in grads.items():
            gk = g * clip_coef
            if params[k].ndim >= 2:
                gk = gk + self.wd * params[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (gk * gk)
            mh = self.m[k] / (1.0 - b1 ** self.t)
            vh = self.v[k] / (1.0 - b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)


class TextCNNClassifier:
    """
    Multi-Scale 1D TextCNN + TF-IDF Dense Fusion Classifier trained via NumPy AdamW.
    """

    def __init__(self, vocab_size: int, tfidf_dim: int, num_classes: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.tfidf_dim = tfidf_dim
        self.num_classes = num_classes

        conv_total = len(CNN_KERNELS) * CNN_FILTERS * 2
        concat_dim = conv_total + TFIDF_PROJ_DIM

        self.params: Dict[str, np.ndarray] = {
            "E": rng.normal(0.0, 0.08, (vocab_size, CNN_EMB_DIM)).astype(np.float32),
            "W_sp": rng.normal(0.0, np.sqrt(2.0 / max(1, tfidf_dim)), (tfidf_dim, TFIDF_PROJ_DIM)).astype(np.float32),
            "b_sp": np.zeros(TFIDF_PROJ_DIM, dtype=np.float32),
            "W_h": rng.normal(0.0, np.sqrt(1.0 / concat_dim), (concat_dim, HIDDEN_DIM)).astype(np.float32),
            "b_h": np.zeros(HIDDEN_DIM, dtype=np.float32),
            "W_cls": rng.normal(0.0, np.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, num_classes)).astype(np.float32),
            "b_cls": np.zeros(num_classes, dtype=np.float32),
        }
        self.params["E"][0] = 0.0

        for k in CNN_KERNELS:
            fan_in = k * CNN_EMB_DIM
            self.params[f"W_c{k}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, CNN_FILTERS)).astype(np.float32)
            self.params[f"b_c{k}"] = np.zeros(CNN_FILTERS, dtype=np.float32)

    def forward(self, seq_ids: np.ndarray, seq_lens: np.ndarray, X_tfidf, train: bool = False, rng=None):
        B = seq_ids.shape[0]
        E = self.params["E"]
        emb = E[seq_ids]  # (B, T, D)
        pad_mask = (seq_ids != 0).astype(np.float32)[:, :, None]
        emb = emb * pad_mask

        pooled_parts = []
        cache = {"seq_ids": seq_ids, "seq_lens": seq_lens, "pad_mask": pad_mask, "X_tfidf": X_tfidf, "conv": {}}

        for k in CNN_KERNELS:
            L = CNN_MAX_LEN - k + 1
            # Vectorized im2col windows: (B, L, k*D)
            windows = np.concatenate([emb[:, i: i + L, :] for i in range(k)], axis=2)
            pre = windows @ self.params[f"W_c{k}"] + self.params[f"b_c{k}"]
            act = np.maximum(0.0, pre)  # ReLU (B, L, F)

            # Mask strictly padded windows (valid positions up to max(1, len - k + 1))
            pos = np.arange(L)[None, :, None]
            valid_win = (pos < np.maximum(1, seq_lens - k + 1)[:, None, None]).astype(np.float32)
            act_masked = act * valid_win

            denom = np.maximum(1.0, (seq_lens - k + 1).astype(np.float32))[:, None]
            mean_pool = np.sum(act_masked, axis=1) / denom

            # For max pool set invalid positions to -1e9 so valid zeros win over padded positions
            act_for_max = np.where(valid_win > 0, act_masked, -1e9)
            argmax_pos = np.argmax(act_for_max, axis=1)  # (B, F)
            max_pool = np.take_along_axis(act_for_max, argmax_pos[:, None, :], axis=1)[:, 0, :]
            max_pool = np.maximum(0.0, max_pool)

            pooled_parts.append(max_pool)
            pooled_parts.append(mean_pool)
            cache["conv"][k] = (windows, act, valid_win, argmax_pos, denom)

        z_cnn = np.concatenate(pooled_parts, axis=1)  # (B, 144)

        z_sp = np.tanh(X_tfidf.dot(self.params["W_sp"]) + self.params["b_sp"])
        z_cat = np.concatenate([z_cnn, z_sp], axis=1)  # (B, 192)

        if train and rng is not None:
            drop_mask = (rng.random(z_cat.shape) > 0.15).astype(np.float32) / 0.85
            z_cat = z_cat * drop_mask
        else:
            drop_mask = None

        h = np.tanh(z_cat @ self.params["W_h"] + self.params["b_h"])
        logits = h @ self.params["W_cls"] + self.params["b_cls"]

        cache.update({"z_sp": z_sp, "z_cat": z_cat, "drop_mask": drop_mask, "h": h})
        return logits, h, cache

    def fit(
        self,
        seq_ids: np.ndarray,
        seq_lens: np.ndarray,
        X_tfidf,
        y: np.ndarray,
        sample_weights: np.ndarray,
        epochs: int = 24,
        batch_size: int = 64,
        lr_max: float = 3.5e-3,
    ):
        N = len(y)
        opt = AdamWOptimizer(self.params, lr=lr_max, weight_decay=1e-4)
        rng = np.random.default_rng(42)

        # Label smoothing target
        smooth = 0.04
        one_hot = np.full((N, self.num_classes), smooth / max(1, self.num_classes - 1), dtype=np.float32)
        one_hot[np.arange(N), y] = 1.0 - smooth

        for epoch in range(epochs):
            perm = rng.permutation(N)
            # Cosine LR schedule
            lr = 3e-4 + 0.5 * (lr_max - 3e-4) * (1.0 + np.cos(np.pi * epoch / max(1, epochs - 1)))

            for start in range(0, N, batch_size):
                idx = perm[start: start + batch_size]
                B = len(idx)
                Xb_ids = seq_ids[idx]
                Xb_lens = seq_lens[idx]
                Xb_tfidf = X_tfidf[idx]
                yb_target = one_hot[idx]
                wb = sample_weights[idx][:, None]

                logits, h, cache = self.forward(Xb_ids, Xb_lens, Xb_tfidf, train=True, rng=rng)
                probs = _softmax(logits)

                # Weighted cross-entropy gradient wrt logits
                d_logits = ((probs - yb_target) * wb) / max(1.0, float(np.sum(wb)))

                grads = {}
                grads["W_cls"] = h.T @ d_logits
                grads["b_cls"] = np.sum(d_logits, axis=0)

                d_h = d_logits @ self.params["W_cls"].T
                d_pre_h = d_h * (1.0 - h * h)
                grads["W_h"] = cache["z_cat"].T @ d_pre_h
                grads["b_h"] = np.sum(d_pre_h, axis=0)

                d_zcat = d_pre_h @ self.params["W_h"].T
                if cache["drop_mask"] is not None:
                    d_zcat = d_zcat * cache["drop_mask"]

                conv_dim = len(CNN_KERNELS) * CNN_FILTERS * 2
                d_zcnn = d_zcat[:, :conv_dim]
                d_zsp = d_zcat[:, conv_dim:]

                # Backprop TF-IDF branch
                d_pre_sp = d_zsp * (1.0 - cache["z_sp"] * cache["z_sp"])
                grads["W_sp"] = Xb_tfidf.T.dot(d_pre_sp).astype(np.float32)
                grads["b_sp"] = np.sum(d_pre_sp, axis=0)

                # Backprop Multi-Scale 1D Convolutions
                d_emb = np.zeros((B, CNN_MAX_LEN, CNN_EMB_DIM), dtype=np.float32)
                offset = 0
                for k in CNN_KERNELS:
                    d_max = d_zcnn[:, offset: offset + CNN_FILTERS]
                    d_mean = d_zcnn[:, offset + CNN_FILTERS: offset + 2 * CNN_FILTERS]
                    offset += 2 * CNN_FILTERS

                    windows, act, valid_win, argmax_pos, denom = cache["conv"][k]
                    L = CNN_MAX_LEN - k + 1
                    d_act = np.zeros_like(act)

                    # Mean pool gradient
                    d_act += (d_mean / denom)[:, None, :] * valid_win

                    # Max pool gradient scatter to argmax positions
                    b_grid = np.arange(B)[:, None]
                    f_grid = np.arange(CNN_FILTERS)[None, :]
                    d_act[b_grid, argmax_pos, f_grid] += d_max

                    # ReLU gradient
                    d_pre = d_act * (act > 0).astype(np.float32) * valid_win

                    # Weight & bias grads over flattened (B*L)
                    win_flat = windows.reshape(-1, k * CNN_EMB_DIM)
                    d_pre_flat = d_pre.reshape(-1, CNN_FILTERS)
                    grads[f"W_c{k}"] = win_flat.T @ d_pre_flat
                    grads[f"b_c{k}"] = np.sum(d_pre_flat, axis=0)

                    d_windows = (d_pre_flat @ self.params[f"W_c{k}"].T).reshape(B, L, k * CNN_EMB_DIM)
                    for j in range(k):
                        d_emb[:, j: j + L, :] += d_windows[:, :, j * CNN_EMB_DIM: (j + 1) * CNN_EMB_DIM]

                d_emb *= cache["pad_mask"]
                d_E = np.zeros_like(self.params["E"])
                np.add.at(d_E, Xb_ids, d_emb)
                d_E[0] = 0.0
                grads["E"] = d_E

                opt.step(self.params, grads, lr)

    def predict(self, seq_ids: np.ndarray, seq_lens: np.ndarray, X_tfidf, batch_size: int = 256):
        all_logits = []
        all_h = []
        for s in range(0, len(seq_ids), batch_size):
            l, h, _ = self.forward(seq_ids[s: s + batch_size], seq_lens[s: s + batch_size], X_tfidf[s: s + batch_size], train=False)
            all_logits.append(l)
            all_h.append(h)
        logits = np.vstack(all_logits)
        hidden = np.vstack(all_h)
        return logits, _softmax(logits), hidden


class RecursiveIntrospectionModel:
    """
    3-Step Recursive Action-Reaction Introspection Network trained with Backpropagation Through Time (BPTT).
    Given latent query representation z^(0) = h_cnn, iteratively updates state across
    (action a^(s), retrieval/browser environment reaction r^(s)) transitions to predict:
      - next action pi^(s+1)
      - epistemic answer sufficiency critic score q^(s+1)
      - domain refinement logit delta^(s+1)
    """

    def __init__(self, num_classes: int, seed: int = 99):
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        in_u = HIDDEN_DIM + NUM_ACTIONS + REACTION_DIM
        in_g = HIDDEN_DIM + REACTION_DIM
        self.params: Dict[str, np.ndarray] = {
            "W_g": rng.normal(0.0, np.sqrt(1.0 / in_g), (in_g, HIDDEN_DIM)).astype(np.float32),
            "b_g": np.full(HIDDEN_DIM, 0.25, dtype=np.float32),
            "W_u": rng.normal(0.0, np.sqrt(1.0 / in_u), (in_u, HIDDEN_DIM)).astype(np.float32),
            "b_u": np.zeros(HIDDEN_DIM, dtype=np.float32),
            "W_pi": rng.normal(0.0, np.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, NUM_ACTIONS)).astype(np.float32),
            "b_pi": np.zeros(NUM_ACTIONS, dtype=np.float32),
            "w_q": rng.normal(0.0, np.sqrt(1.0 / HIDDEN_DIM), (HIDDEN_DIM, 1)).astype(np.float32),
            "b_q": np.zeros(1, dtype=np.float32),
            "W_delta": rng.normal(0.0, 0.04, (HIDDEN_DIM, num_classes)).astype(np.float32),
            "b_delta": np.zeros(num_classes, dtype=np.float32),
        }

    def step_forward(self, z: np.ndarray, action_oh: np.ndarray, reaction: np.ndarray):
        in_g = np.concatenate([z, reaction], axis=1)
        g = _sigmoid(in_g @ self.params["W_g"] + self.params["b_g"])
        in_u = np.concatenate([z, action_oh, reaction], axis=1)
        u = np.tanh(in_u @ self.params["W_u"] + self.params["b_u"])
        z_next = g * z + (1.0 - g) * u

        pi_logits = z_next @ self.params["W_pi"] + self.params["b_pi"]
        q_hat = _sigmoid(z_next @ self.params["w_q"] + self.params["b_q"])
        delta = z_next @ self.params["W_delta"] + self.params["b_delta"]
        cache = (z, in_g, g, in_u, u, z_next)
        return z_next, pi_logits, q_hat, delta, cache

    def fit_bptt(
        self,
        H_hidden: np.ndarray,
        logits0: np.ndarray,
        y_domain: np.ndarray,
        class_tags: List[str],
        epochs: int = 35,
        batch_size: int = 64,
    ):
        """
        Synthesizes 2,400+ grounded 3-step Action-Reaction rollouts over real query embeddings
        and trains recursive introspection via unrolled BPTT.
        Reaction dims (12):
          [0] bm25_top_rel   [1] span_share    [2] token_cov      [3] domain_agree
          [4] clf_top_prob   [5] clf_entropy   [6] passage_sat    [7] brief_cue
          [8] detail_cue     [9] compare_cue   [10] web_srcs      [11] text_richness
        """
        rng = np.random.default_rng(123)
        N_real = len(H_hidden)
        M = max(2400, N_real)
        steps = 3

        pick = rng.integers(0, N_real, size=M)
        z0 = H_hidden[pick].copy()
        base_logits = logits0[pick].copy()
        dom_y = y_domain[pick].copy()

        # Allocate 3-step rollout buffers
        acts = np.zeros((steps, M, NUM_ACTIONS), dtype=np.float32)
        reacts = np.zeros((steps, M, REACTION_DIM), dtype=np.float32)
        tgt_act = np.zeros((steps, M), dtype=np.int32)
        tgt_q = np.zeros((steps, M, 1), dtype=np.float32)

        conv_indices = {i for i, t in enumerate(class_tags) if t in ("greeting", "thanks_goodbye", "language_support", "wer_evaluation", "knowledge_auto_upgrade")}

        for i in range(M):
            is_conv = int(dom_y[i]) in conv_indices
            scenario = int(rng.integers(0, 6))
            brief = 1.0 if scenario == 4 and rng.random() < 0.7 else 0.0
            detail = 1.0 if (scenario in (1, 3) and not brief) else 0.0
            compare = 1.0 if scenario == 5 else 0.0

            if is_conv:
                # Scenario: Conversational query needs direct reply
                r0 = [0.1, 0.0, 0.1, 1.0, 0.88, 0.15, 0.9, brief, 0.0, 0.0, 0.0, 0.7]
                acts[0, i, 0] = 1.0
                reacts[0, i] = r0
                tgt_act[0, i] = 0
                tgt_q[0, i, 0] = 0.94

                acts[1, i, 0] = 1.0
                reacts[1, i] = r0
                tgt_act[1, i] = 0
                tgt_q[1, i, 0] = 0.95

                acts[2, i, 0] = 1.0
                reacts[2, i] = r0
                tgt_act[2, i] = 0
                tgt_q[2, i, 0] = 0.96
            elif scenario == 5:
                # Comparison query -> local check -> web enrich missing side -> compare_entities
                acts[0, i, 2] = 1.0
                reacts[0, i] = [0.42, 0.35, 0.45, 0.6, 0.65, 0.40, 0.4, 0.0, detail, 1.0, 0.0, 0.4]
                tgt_act[0, i] = 3  # browse_web_multi to fetch missing side

                acts[1, i, 3] = 1.0
                reacts[1, i] = [0.78, 0.72, 0.80, 0.9, 0.75, 0.28, 0.8, 0.0, detail, 1.0, 0.66, 0.85]
                tgt_act[1, i] = 4  # compare_entities
                tgt_q[1, i, 0] = 0.86

                acts[2, i, 4] = 1.0
                reacts[2, i] = [0.85, 0.80, 0.85, 1.0, 0.84, 0.20, 0.9, 0.0, detail, 1.0, 0.80, 0.90]
                tgt_act[2, i] = 4
                tgt_q[2, i, 0] = 0.93
            elif scenario in (0, 4):
                # Strong local KB match satisfies standard/brief query
                acts[0, i, 2] = 1.0
                reacts[0, i] = [0.86 + rng.uniform(-0.08, 0.08), 0.78, 0.82, 1.0, 0.85, 0.18, 0.88, brief, 0.0, 0.0, 0.2, 0.75]
                tgt_act[0, i] = 2
                tgt_q[0, i, 0] = 0.90

                acts[1, i, 2] = 1.0
                reacts[1, i] = reacts[0, i]
                tgt_act[1, i] = 2
                tgt_q[1, i, 0] = 0.92

                acts[2, i, 2] = 1.0
                reacts[2, i] = reacts[0, i]
                tgt_act[2, i] = 2
                tgt_q[2, i, 0] = 0.93
            elif scenario in (1, 2):
                # Weak local KB OR detailed query lacking depth -> trigger multi-website browse -> deep synthesis
                acts[0, i, 2] = 1.0
                weak_rel = 0.24 if scenario == 2 else 0.58
                weak_rich = 0.25 if detail else 0.20
                reacts[0, i] = [weak_rel, 0.22, 0.30, 0.4, 0.52, 0.62, 0.25, 0.0, detail, 0.0, 0.0, weak_rich]
                tgt_act[0, i] = 3  # browse_web_multi
                tgt_q[0, i, 0] = 0.24

                acts[1, i, 3] = 1.0
                reacts[1, i] = [0.76, 0.70, 0.78, 0.9, 0.78, 0.25, 0.82, 0.0, detail, 0.0, 0.75, 0.82]
                tgt_act[1, i] = 5 if detail else 3
                tgt_q[1, i, 0] = 0.84

                acts[2, i, tgt_act[1, i]] = 1.0
                reacts[2, i] = [0.88, 0.82, 0.88, 1.0, 0.88, 0.16, 0.92, 0.0, detail, 0.0, 0.90, 0.94]
                tgt_act[2, i] = 5 if detail else 2
                tgt_q[2, i, 0] = 0.94
            else:
                # Math evaluation reaction
                acts[0, i, 1] = 1.0
                reacts[0, i] = [0.95, 0.95, 0.95, 1.0, 0.95, 0.05, 1.0, 1.0, 0.0, 0.0, 0.0, 0.8]
                tgt_act[0, i] = 1
                tgt_q[0, i, 0] = 0.98
                acts[1, i, 1] = 1.0
                reacts[1, i] = reacts[0, i]
                tgt_act[1, i] = 1
                tgt_q[1, i, 0] = 0.98
                acts[2, i, 1] = 1.0
                reacts[2, i] = reacts[0, i]
                tgt_act[2, i] = 1
                tgt_q[2, i, 0] = 0.99

        opt = AdamWOptimizer(self.params, lr=4e-3, weight_decay=1e-4)
        final_q_acc = 0.0
        final_act_acc = 0.0

        for epoch in range(epochs):
            perm = rng.permutation(M)
            lr = 4e-4 + 0.5 * (4e-3 - 4e-4) * (1.0 + np.cos(np.pi * epoch / max(1, epochs - 1)))

            for s_idx in range(0, M, batch_size):
                idx = perm[s_idx: s_idx + batch_size]
                B = len(idx)
                z = z0[idx]
                l0 = base_logits[idx]
                yd = dom_y[idx]
                yd_oh = np.eye(self.num_classes, dtype=np.float32)[yd]

                step_caches = []
                step_outs = []
                for s in range(steps):
                    z, pi_logits, q_hat, delta, cache = self.step_forward(z, acts[s, idx], reacts[s, idx])
                    step_caches.append(cache)
                    step_outs.append((pi_logits, q_hat, delta))

                # Unrolled BPTT backward pass
                grads = {k: np.zeros_like(v) for k, v in self.params.items()}
                dz_next = np.zeros((B, HIDDEN_DIM), dtype=np.float32)

                for s in range(steps - 1, -1, -1):
                    z_prev, in_g, g, in_u, u, z_next = step_caches[s]
                    pi_logits, q_hat, delta = step_outs[s]

                    pi_prob = _softmax(pi_logits)
                    pi_tgt = np.eye(NUM_ACTIONS, dtype=np.float32)[tgt_act[s, idx]]
                    d_pi = (pi_prob - pi_tgt) / B

                    d_q_logit = (q_hat - tgt_q[s, idx]) * 1.5 / B

                    dom_prob = _softmax(l0 + delta)
                    d_delta = (dom_prob - yd_oh) * 0.45 / B

                    grads["W_pi"] += z_next.T @ d_pi
                    grads["b_pi"] += np.sum(d_pi, axis=0)
                    grads["w_q"] += z_next.T @ d_q_logit
                    grads["b_q"] += np.sum(d_q_logit, axis=0)
                    grads["W_delta"] += z_next.T @ d_delta
                    grads["b_delta"] += np.sum(d_delta, axis=0)

                    dz = (
                        dz_next
                        + d_pi @ self.params["W_pi"].T
                        + d_q_logit @ self.params["w_q"].T
                        + d_delta @ self.params["W_delta"].T
                    )

                    # z_next = g * z_prev + (1 - g) * u
                    du = dz * (1.0 - g)
                    dg = dz * (z_prev - u)
                    dz_prev_direct = dz * g

                    d_pre_u = du * (1.0 - u * u)
                    d_pre_g = dg * g * (1.0 - g)

                    grads["W_u"] += in_u.T @ d_pre_u
                    grads["b_u"] += np.sum(d_pre_u, axis=0)
                    grads["W_g"] += in_g.T @ d_pre_g
                    grads["b_g"] += np.sum(d_pre_g, axis=0)

                    dz_next = (
                        dz_prev_direct
                        + (d_pre_u @ self.params["W_u"].T)[:, :HIDDEN_DIM]
                        + (d_pre_g @ self.params["W_g"].T)[:, :HIDDEN_DIM]
                    )

                opt.step(self.params, grads, lr)

        # Evaluate calibration after training
        z = z0
        for s in range(steps):
            z, pi_logits, q_hat, _, _ = self.step_forward(z, acts[s], reacts[s])
            if s == steps - 1:
                final_act_acc = float(np.mean(np.argmax(pi_logits, axis=1) == tgt_act[s]))
                final_q_acc = float(1.0 - np.mean(np.abs(q_hat[:, 0] - tgt_q[s, :, 0])))

        return {
            "rollout_trajectories": int(M),
            "introspection_steps": steps,
            "policy_action_accuracy": round(final_act_acc, 4),
            "critic_calibration_score": round(final_q_acc, 4),
        }
