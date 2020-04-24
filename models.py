import torch
from torch import nn
import numpy as np
import math, copy, time
import torch.nn.functional as F
from collections import defaultdict
from embeddings import PositionalEncoding, Embeddings
from utils.pad import pad_masking, subsequent_masking

PAD_TOKEN_ID = 0


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
        return x + self.dropout(sublayer(self.norm(x)))


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads_count, d_model, dropout_prob=0.1, mode='self-attention'):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')

        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode

        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)

        self.attention = None
        self.dropout = nn.Dropout(p=dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        # For cache
        self.key_projected = None
        self.value_projected = None

    # def attention(self, query, key, value, mask=None, dropout=None):
    #     "Compute 'Scaled Dot Product Attention'"
    #     d_k = query.size(-1)
    #     scores = torch.matmul(query, key.transpose(-2, -1)) \
    #              / math.sqrt(d_k)
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #     p_attn = F.softmax(scores, dim=-1)
    #     if dropout is not None:
    #         p_attn = dropout(p_attn)
    #     return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None, layer_cache=None):
        """
        Args:
            :param query: (batch_size, query_len, model_dim)
            :param key: (batch_size, key_len, model_dim)
            :param value: (batch_size, value_len, model_dim)
            :param mask: (batch_size, query_len, key_len)
            :param layer_cache:
            :state DecoderState:
        """
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.heads_count

        query_projected = self.query_projection(query)
        # print('query_projected', query_projected.shape)
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


class PointwiseFeedForwardNetwork(nn.Module):
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

    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

        self.feed_forward = nn.Sequential(self.w_1, self.dropout, self.relu, self.w_2, self.dropout)

        # Solution 2: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        self.feed_forward_v2 = nn.Sequential(self.w_1, self.relu, self.dropout, self.w_2)

    def forward(self, x):
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, layer, N, d_model, src_vocab_size):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        args:
           x: embedded_sequence, (batch_size, seq_len, embed_size)
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    # def __init__(self, d_model, heads_count, d_ff, dropout_prob):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, d_model)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.dropout(x)  # Optional
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N, d_model, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        # Generator Solution 1
        self.generator = Generator(d_model, tgt_vocab_size)

        # Generator Solution 2
        # self.generator = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)
        # self.generator.weight = self.embedding.weight

    def forward(self, x, memory, src_mask, tgt_mask=None, state=None):
        # x: (batch_size, seq_len - 1, d_model)
        # memory: (batch_size, seq_len, d_model)

        # if state is not None:
        #     x = torch.cat([state.previous_inputs, x], dim=1)
        #
        #     state.previous_inputs = x

        for layer_index, decoder_layer in enumerate(self.layers):
            if state is None:
                x = decoder_layer(x, memory, src_mask, tgt_mask)
            else:  # Use cache
                layer_cache = state.layer_caches[layer_index]
                # print('inputs_mask', inputs_mask)
                x = decoder_layer(x, memory, src_mask, tgt_mask, layer_cache)

                state.update_state(
                    layer_index=layer_index,
                    layer_mode='self-attention',
                    key_projected=decoder_layer.self_attn.sublayer.key_projected,
                    value_projected=decoder_layer.self_attn.sublayer.value_projected,
                )
                state.update_state(
                    layer_index=layer_index,
                    layer_mode='memory-attention',
                    key_projected=decoder_layer.src_attn.sublayer.key_projected,
                    value_projected=decoder_layer.src_attn.sublayer.value_projected,
                )

        generated = self.generator(x)  # (batch_size, seq_len, vocab_size)
        # generated = self.norm(x)
        return generated, state

    def init_decoder_state(self, **args):
        return DecoderState()


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    """
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_cache=None):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class DecoderState:
    def __init__(self):
        self.previous_inputs = torch.tensor([])
        self.layer_caches = defaultdict(lambda: {'self-attention': None, 'memory-attention': None})

    def update_state(self, layer_index, layer_mode, key_projected, value_projected):
        self.layer_caches[layer_index][layer_mode] = {
            'key_projected': key_projected,
            'value_projected': value_projected
        }

    # def repeat_beam_size_times(self, beam_size):
    #     self.
    #     self.src = self.src.data.repeat(beam_size, 1)

    def beam_update(self, positions):
        for layer_index in self.layer_caches:
            for mode in ('self-attention', 'memory-attention'):
                if self.layer_caches[layer_index][mode] is not None:
                    for projection in self.layer_caches[layer_index][mode]:
                        cache = self.layer_caches[layer_index][mode][projection]
                        if cache is not None:
                            cache.data.copy_(cache.data.index_select(0, positions))


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, sources, inputs):
        # sources : (batch_size, sources_len)
        # inputs : (batch_size, targets_len - 1)
        batch_size, sources_len = sources.size()
        batch_size, inputs_len = inputs.size()

        sources_mask = pad_masking(sources, sources_len)
        memory_mask = pad_masking(sources, inputs_len)
        inputs_mask = subsequent_masking(inputs) | pad_masking(inputs, inputs_len)

        # (batch_size, seq_len, d_model)
        memory = self.encoder(self.src_embed(sources), sources_mask)  # Context Vectors

        # (batch_size, seq_len, d_model)
        outputs, state = self.decoder(self.tgt_embed(inputs), memory, memory_mask, inputs_mask)
        return outputs


# N=6,
# d_model=512,
# d_ff=2048,
# h=8,
# dropout=0.1
def build_model(config, src_vocab_size, tgt_vocab_size):
    c = copy.deepcopy

    N = config['layers_count']
    d_model = config['d_model']
    d_ff = config['d_ff']
    h = config['heads_count']
    dropout = config['dropout_prob']

    attn = MultiHeadedAttention(h, d_model)
    ff = PointwiseFeedForwardNetwork(d_ff, d_model, dropout)
    position = PositionalEncoding(d_model, dropout)

    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout),  # encode layer
                      N,  # nums of layers in encode
                      d_model,
                      src_vocab_size)  # Dim of vector

    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),
                      N,  # nums of layers in encode
                      d_model,
                      tgt_vocab_size)  # Dim of vector

    model = Transformer(encoder,
                        decoder,
                        nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
                        nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
                        Generator(d_model, tgt_vocab_size))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
