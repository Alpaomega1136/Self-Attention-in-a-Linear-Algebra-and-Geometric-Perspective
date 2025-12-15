import json
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent  
MAX_SEQ_LEN = 256
D_MODEL = 256
N_HEADS = 4
VOCAB_PATH = BASE_DIR / "models" / "imdb_vocab.json"
WEIGHTS_PATH = BASE_DIR / "models" / "imdb_self_attn_best.pt"


def simple_tokenize(text: str) -> List[str]:
    text = text.lower().replace("<br />", " ")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def text_to_ids_and_mask(text: str, vocab: dict, max_len: int) -> Tuple[List[int], List[int]]:
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
    mask = [1] * len(ids)
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids += [vocab["<pad>"]] * pad_len
        mask += [0] * pad_len
    return ids, mask


def load_vocab(path: Path = VOCAB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ScaledDotProductSelfAttention(nn.Module):
    """
    Blok multihead attention + FFN dengan residual, LayerNorm, dropout.
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self, d_model=256, n_heads=4, p_drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.dropout_attn = nn.Dropout(p_drop)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout_ff = nn.Dropout(p_drop)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, X, key_padding_mask=None):
        # Linear projections di MultiheadAttention → Q, K, V
        # S = Q K^T / sqrt(d_k); A = softmax(S); Z = A V
        Z, A = self.attn(
            X,
            X,
            X,
            key_padding_mask=key_padding_mask,  # True untuk PAD
            need_weights=True,
            average_attn_weights=False,
        )
        X = self.ln1(X + self.dropout_attn(Z))
        FF = self.ff(X)
        X = self.ln2(X + self.dropout_ff(FF))
        return X, A


class IMDBSelfAttnClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_len=256, n_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.block1 = ScaledDotProductSelfAttention(d_model=d_model, n_heads=n_heads, p_drop=0.1)
        self.block2 = ScaledDotProductSelfAttention(d_model=d_model, n_heads=n_heads, p_drop=0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        X = self.token_emb(input_ids) + self.pos_emb(positions)  # X: embedding + posisi

        key_padding = attention_mask == 0  # True untuk PAD
        A_list = []
        X, A1 = self.block1(X, key_padding_mask=key_padding)
        A_list.append(A1)
        X, A2 = self.block2(X, key_padding_mask=key_padding)
        A_list.append(A2)

        mask = attention_mask.unsqueeze(-1)
        summed = (X * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / lengths

        logits = self.classifier(pooled).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, logits, A_list[-1]


def load_model(vocab, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMDBSelfAttnClassifier(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        max_len=MAX_SEQ_LEN,
        n_heads=N_HEADS,
    ).to(device)
    if WEIGHTS_PATH.exists():
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model, device


def predict_with_self_attention(model, vocab, text: str, device=None):
    """
    Mengembalikan (probabilitas positif, tokens, attention_matrix head-0 dipotong non-pad).
    """
    if device is None:
        device = next(model.parameters()).device
    ids, mask = text_to_ids_and_mask(text, vocab, MAX_SEQ_LEN)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs, logits, A = model(input_ids, attention_mask)
        prob_pos = probs[0].item()
        tokens = [tok for tok, m in zip(simple_tokenize(text)[:MAX_SEQ_LEN], mask) if m == 1]
        A_matrix = A[0, 0].cpu().numpy()  # head 0
        A_cut = A_matrix[: len(tokens), : len(tokens)]
    return prob_pos, tokens, A_cut
