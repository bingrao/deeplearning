import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """
    Produce N identical layers.
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    (See citation for details: https://arxiv.org/abs/1607.06450).
    """
    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        # print(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    (See citation for details: https://arxiv.org/abs/1512.03385).
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        # return x + self.dropout(sublayer(self.norm(x)))
        reg = self.norm((x + self.dropout(sublayer(x))))
        return reg


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class PositionwiseFeedForward(nn.Module):
    """
    In addition to attention sub-layers, each of the layers in our encoder and decoder contains
    a fully connected feed-forward network, which is applied to each position separately and
    identically. This consists of two linear transformations with a ReLU activation in between.

    While the linear transformations are the same across different positions, they use different
    parameters from layer to layer. Another way of describing this is as two convolutions with kernel
    size 1. The dimensionality of input and output is $d_{\text{model}}=512$, and the inner-layer
    has dimensionality $d_{ff}=2048$.

    Implements FFN equation.
    """

    def __init__(self, ctx, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.context = ctx
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.feed_forward = nn.Sequential(self.w_1, self.dropout, self.relu, self.w_2, self.dropout)

        # Solution 2: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        self.feed_forward_v1 = nn.Sequential(self.w_1, self.relu, self.dropout, self.w_2)

    def forward(self, x):
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)
        # return self.w_2(self.dropout(F.relu(self.w_1(x))))


