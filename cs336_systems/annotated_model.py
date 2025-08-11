from __future__ import annotations

import math

import torch
from cs336_basics.nn_utils import softmax
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_systems.profiling_utils import profiling_range


def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    d_k = K.shape[-1]
    with profiling_range("attention_scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with profiling_range("attention_weights"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with profiling_range("output"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return output


def annotated_lm_forward(
    self, x: Int[Tensor, " ... sequence_length"]
) -> Float[Tensor, " ... sequence_length vocab_size"]:
    """
    Args:
        x: Input IDs for language modeling.

    Returns: A FloatTensor of shape
        (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
        distribution for each token.
    """
    _, sequence_length = x.size()

    # (batch size, sequence_length, d_model)
    with profiling_range("token_embeddings"):
        x = self.token_embeddings(x)

    for layer in self.layers:
        # (batch size, sequence_length, d_model)
        with profiling_range("layer"):
            x = layer(x)

    # (batch size, sequence_length, d_model)
    with profiling_range("ln_final"):
        x = self.ln_final(x)

    # (batch size, sequence_length, vocab_size)
    with profiling_range("lm_head"):
        x = self.lm_head(x)

    return x


def annotated_transformer_block_forward(self, x: torch.Tensor):
    """
    Args:
        x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
            The input to process with the Transformer block.

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """
    # NOTE: this is a pre-norm Transformer, and differs from the original
    # description in the paper.
    # Apply the multi-head self-attention sublayer
    with profiling_range("attn"):
        x_attn = self.attn(self.ln1(x))

    attn_sublayer_output = x + x_attn

    # Apply the feed-forward sublayer
    with profiling_range("ffn"):
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))

    ffn_sublayer_output = attn_sublayer_output + x_ffn
    return ffn_sublayer_output
