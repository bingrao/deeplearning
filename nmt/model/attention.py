import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nmt.model.common import clones


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttentionWithMetrics(nn.Module):

    def __init__(self, ctx, heads_count, d_model, dropout_prob=0.1, mode='self-attention'):
        super(MultiHeadAttentionWithMetrics, self).__init__()
        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')
        self.context = ctx
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None
        # For cache
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        self.context.logger.debug('[%s] attention mask %s', self.__class__.__name__, mask)
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.heads_count

        query_projected = self.query_projection(query)
        self.context.logger.debug('[%s] query_projected %s', self.__class__.__name__, query_projected.shape)

        if layer_cache is None or layer_cache[self.mode] is None:  # Don't use cache
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:  # Use cache
            if self.mode == 'self-attention':
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)

                key_projected = torch.cat([key_projected, layer_cache[self.mode]['key_projected']], dim=1)
                value_projected = torch.cat([value_projected, layer_cache[self.mode]['value_projected']], dim=1)
            elif self.mode == 'memory-attention':
                key_projected = layer_cache[self.mode]['key_projected']
                value_projected = layer_cache[self.mode]['value_projected']

        # For cache
        self.key_projected = key_projected
        self.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        # (batch_size, heads_count, query_len, d_head)
        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
        # print('query_heads', query_heads.shape)
        # print(batch_size, key_len, self.heads_count, d_head)
        # print(key_projected.shape)
        # (batch_size, heads_count, key_len, d_head)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1, 2)

        # (batch_size, heads_count, value_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1, 2)

        # (batch_size, heads_count, query_len, key_len)
        attention_weights = self.scaled_dot_product(query_heads, key_heads)

        if mask is not None:
            # print('mode', self.mode)
            # print('mask', mask.shape)
            # print('attention_weights', attention_weights.shape)
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e18)

        self.attention = self.softmax(attention_weights)  # Save attention to the object
        # print('attention_weights', attention_weights.shape)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)  # (batch_size, heads_count, query_len, d_head)
        # print('context_heads', context_heads.shape)
        context_sequence = context_heads.transpose(1, 2).contiguous()  # (batch_size, query_len, heads_count, d_head)
        context = context_sequence.view(batch_size, query_len, d_model)  # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        # print('final_output', final_output.shape)
        self.context.logger.debug("[%s] The query %s, key %s, value %s, final_output %s dimension",
            self.__class__.__name__, query.size(), key.size(), value.size(), final_output.size())
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """

        Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)  # (batch_size, heads_count, query_len, key_len)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, ctx, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.context = ctx
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)
        self.belong = "transformer"

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [ln(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for ln, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # [MultiHeadedAttention] The query torch.Size([10, 8, 34, 16]),
        #                        key torch.Size([10, 8, 34, 16]),
        #                        value torch.Size([10, 8, 34, 16]),
        #                        output torch.Size([10, 34, 128])
        self.context.logger.debug("[%s] The query %s, key %s, value %s, output %s", self.__class__.__name__,
                                  query.size(), key.size(), value.size(), x.size())

        return self.linears[-1](x)
