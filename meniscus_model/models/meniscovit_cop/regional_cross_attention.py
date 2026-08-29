"""Sagittal-coronal regional cross-attention in real coordinate space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RegionalCrossAttentionInfo:
    """Optional diagnostics for regional cross-attention.

    Attributes:
        attention_weights: Tensor ``[B, num_heads, Nq, k]``.
        neighbor_indices: LongTensor ``[B, Nq, k]`` into the key/value tokens.
    """

    attention_weights: torch.Tensor
    neighbor_indices: torch.Tensor


class RegionalCrossAttention(nn.Module):
    """Cross-attend from query tokens to nearest spatial tokens in another plane.

    Inputs:
        ``query_tokens``: Tensor ``[B, Nq, C]``.
        ``query_coords``: Tensor ``[B, Nq, 3]``.
        ``key_value_tokens``: Tensor ``[B, Nk, C]``.
        ``key_value_coords``: Tensor ``[B, Nk, 3]``.

    Operation:
        1. Compute real-world distances with ``torch.cdist``: ``[B,Nq,Nk]``.
        2. Keep only nearest ``k`` key/value tokens for each query: ``[B,Nq,k]``.
        3. Run multi-head attention over those local key/value tokens only.

    Output:
        ``fused_query_tokens``: Tensor ``[B, Nq, C]``.
        If ``return_attention=True``, returns ``(fused_query_tokens, info)``.

    This first implementation intentionally uses ``torch.cdist + topk`` for
    clarity. It avoids full attention logits ``[B,heads,Nq,Nk]``; a future KNN
    backend can replace only the neighbor search.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_nearest: int = 32,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if k_nearest <= 0:
            raise ValueError("k_nearest must be positive")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_nearest = k_nearest
        self.scale = self.head_dim**-0.5

        self.query_norm = norm_layer(embed_dim)
        self.key_value_norm = norm_layer(embed_dim)
        self.out_norm = norm_layer(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def nearest_k_indices(query_coords: torch.Tensor, key_value_coords: torch.Tensor, k: int) -> torch.Tensor:
        """Find nearest key/value token indices by Euclidean world distance.

        Args:
            query_coords: Tensor ``[B,Nq,3]``.
            key_value_coords: Tensor ``[B,Nk,3]``.
            k: Number of neighbors.

        Returns:
            LongTensor ``[B,Nq,min(k,Nk)]``.
        """

        if query_coords.ndim != 3 or key_value_coords.ndim != 3:
            raise ValueError("coords must be shaped [B,N,3]")
        if query_coords.shape[0] != key_value_coords.shape[0] or query_coords.shape[-1] != 3 or key_value_coords.shape[-1] != 3:
            raise ValueError("query_coords and key_value_coords must be [B,N,3] with matching B")
        distances = torch.cdist(query_coords.float(), key_value_coords.float(), p=2)  # [B,Nq,Nk]
        return torch.topk(distances, k=min(k, key_value_coords.shape[1]), dim=-1, largest=False).indices

    def _validate_inputs(
        self,
        query_tokens: torch.Tensor,
        query_coords: torch.Tensor,
        key_value_tokens: torch.Tensor,
        key_value_coords: torch.Tensor,
    ) -> None:
        if query_tokens.ndim != 3 or key_value_tokens.ndim != 3:
            raise ValueError("tokens must be shaped [B,N,C]")
        if query_tokens.shape[0] != key_value_tokens.shape[0]:
            raise ValueError("query_tokens and key_value_tokens must have the same batch size")
        if query_tokens.shape[-1] != self.embed_dim or key_value_tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"token embedding dim must be {self.embed_dim}")
        if query_coords.shape[:2] != query_tokens.shape[:2] or query_coords.shape[-1] != 3:
            raise ValueError("query_coords must be shaped [B,Nq,3]")
        if key_value_coords.shape[:2] != key_value_tokens.shape[:2] or key_value_coords.shape[-1] != 3:
            raise ValueError("key_value_coords must be shaped [B,Nk,3]")

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_coords: torch.Tensor,
        key_value_tokens: torch.Tensor,
        key_value_coords: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RegionalCrossAttentionInfo]:
        self._validate_inputs(query_tokens, query_coords, key_value_tokens, key_value_coords)
        batch_size, num_query, channels = query_tokens.shape
        num_key_value = key_value_tokens.shape[1]

        neighbor_indices = self.nearest_k_indices(query_coords, key_value_coords, self.k_nearest)
        local_k_count = neighbor_indices.shape[-1]

        q_tokens = self.query_norm(query_tokens)
        kv_tokens = self.key_value_norm(key_value_tokens)

        # Q: [B,Nq,C] -> [B,H,Nq,D]
        q = self.q(q_tokens).view(batch_size, num_query, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V before neighbor gather: [B,Nk,C].
        all_k = self.k(kv_tokens)
        all_v = self.v(kv_tokens)

        # Gather local K/V only. Output [B,Nq,k,C], never [B,H,Nq,Nk].
        gather_index = neighbor_indices.unsqueeze(-1).expand(batch_size, num_query, local_k_count, channels)
        expanded_k = all_k.unsqueeze(1).expand(batch_size, num_query, num_key_value, channels)
        expanded_v = all_v.unsqueeze(1).expand(batch_size, num_query, num_key_value, channels)
        local_k = torch.gather(expanded_k, dim=2, index=gather_index)
        local_v = torch.gather(expanded_v, dim=2, index=gather_index)

        # Local K/V: [B,Nq,k,C] -> [B,H,Nq,k,D].
        local_k = local_k.view(batch_size, num_query, local_k_count, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        local_v = local_v.view(batch_size, num_query, local_k_count, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Attention logits: [B,H,Nq,k].
        attn_logits = (q.unsqueeze(3) * local_k).sum(dim=-1) * self.scale
        attention_weights = attn_logits.softmax(dim=-1)
        dropped_attention = self.attn_drop(attention_weights)

        # Weighted sum over k: [B,H,Nq,D] -> [B,Nq,C].
        fused = (dropped_attention.unsqueeze(-1) * local_v).sum(dim=3)
        fused = fused.transpose(1, 2).reshape(batch_size, num_query, channels)
        fused = self.proj_drop(self.proj(fused))

        fused_query_tokens = self.out_norm(query_tokens + fused)
        if return_attention:
            info = RegionalCrossAttentionInfo(
                attention_weights=attention_weights,
                neighbor_indices=neighbor_indices,
            )
            return fused_query_tokens, info
        return fused_query_tokens
