import math
import random
import torch
from torch import nn
from torch.nn import functional as F


def get_device():
    """Select available device: cuda > mps > xpu > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class AddPositionalEncoding(nn.Module):
    """
    Adds sine-based positional encoding to embeddings.
    """
    def __init__(self, max_len: int, model_dim: int):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / model_dim)
        )
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :]


class QKVAttention(nn.Module):
    """
    Multi-head QKV attention with optional past-key/value caching.
    """
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.causal = causal

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.o_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor = None,
        past_v: torch.Tensor = None,
    ):
        # x: [batch, seq, model_dim]
        B, T, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for heads: [batch, heads, seq, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # concatenate past kv
        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.tril(torch.ones(T, k.size(2), device=x.device)).unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # attention output
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)

        # return output and new kv
        return out, k, v


class TransformerBlock(nn.Module):
    """
    Single transformer block: attention + feed-forward.
    """
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = QKVAttention(model_dim, num_heads, dropout, causal)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_k: torch.Tensor = None,
        past_v: torch.Tensor = None,
    ):
        # Attention with residual
        y, new_k, new_v = self.attn(self.ln1(x), past_k, past_v)
        x = x + y
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        return x, new_k, new_v


class PicoGPT(nn.Module):
    """
    Very small GPT-like model with KV-cache support for fast inference.
    """
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_enc = AddPositionalEncoding(max_seq_len, model_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, hidden_dim, dropout, causal=True)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)

        # initialize
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        for p in self.head.parameters():
            nn.init.zeros_(p)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for all positions.
        idx: [batch, seq]
        """
        x = self.token_emb(idx)
        x = self.pos_enc(x)
        x = self.drop(x)
        for block in self.blocks:
            x, _, _ = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self,
        idx: torch.Tensor,
        max_new_tokens: int,
        eos_token: int = None
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache:
          idx: [batch, seq]
          returns: [batch, seq + max_new_tokens]
        """
        self.eval()
        device = idx.device
        batch_size, seq_len = idx.size()

        # initialize cache
        past_ks = [None] * len(self.blocks)
        past_vs = [None] * len(self.blocks)

        generated = idx
        for _ in range(max_new_tokens):
            # only feed the last token
            x = generated[:, -1:].clone()
            x = self.token_emb(x)
            # positional encode only last position
            pos = seq_len
            pe = self.pos_enc.pe[pos : pos + 1].unsqueeze(0)
            x = x + pe
            x = self.drop(x)

            # forward through blocks with cache
            for i, block in enumerate(self.blocks):
                x, new_k, new_v = block(x, past_ks[i], past_vs[i])
                past_ks[i] = new_k
                past_vs[i] = new_v

            x = self.ln_f(x)
            logits = self.head(x)  # [batch, 1, vocab]
            next_token = torch.argmax(logits[..., 0, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            seq_len += 1

            if eos_token is not None and (next_token == eos_token).any():
                break

        return generated

    def compute_loss(self, idx: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        # shift tokens for causal
        return F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                               idx[..., 1:].reshape(-1))


# Example usage
if __name__ == "__main__":
    device = get_device()
    # toy data: reverse sequence after '>'
    def generate_data(n, L=25):
        symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ>")
        data = []
        for _ in range(n):
            seq = [random.choice(symbols[:-1]) for _ in range(L)]
            s = seq + ['>'] + seq[::-1]
            data.append(torch.tensor([ord(c) % 256 for c in s], dtype=torch.long))
        return torch.stack(data)

    data = generate_data(1024, 25).to(device)
    model = PicoGPT(vocab_size=256, max_seq_len=100).to(device)
    # training and generation loops omitted for brevity
