import torch
from torch import nn
import numpy as np
import math, copy, time
import torch.nn.functional as F
from collections import defaultdict
from embeddings import PositionalEncoding
from utils.pad import pad_masking, subsequent_masking
PAD_TOKEN_ID = 0


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    (See citation for details: https://arxiv.org/abs/1607.06450).
    """
    def __init__(self, features_count, epsilon=1e-6):
        super(LayerNorm, self).__init__()

        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
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
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Sublayer(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, embedding):
        super(Generator, self).__init__()
        self.proj = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)
        self.weight = embedding.weight

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class MultiHeadedAttention(nn.Module):

    def __init__(self, heads_count, d_model, dropout_prob, mode='self-attention'):
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
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None
        # For cache
        self.key_projected = None
        self.value_projected = None

    def attention(query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None, layer_cache=None):
        """

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, value_len, model_dim)
            mask: (batch_size, query_len, key_len)
            state: DecoderState
        """
        # print('attention mask', mask)
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

        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1,
                                                                                                      2)  # (batch_size, heads_count, query_len, d_head)
        # print('query_heads', query_heads.shape)
        # print(batch_size, key_len, self.heads_count, d_head)
        # print(key_projected.shape)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1,
                                                                                                2)  # (batch_size, heads_count, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1,
                                                                                                      2)  # (batch_size, heads_count, value_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads,
                                                    key_heads)  # (batch_size, heads_count, query_len, key_len)

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

    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)


class Encoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )

    def forward(self, sources, mask):
        """
        args:
           sources: embedded_sequence, (batch_size, seq_len, embed_size)
        """
        sources = self.embedding(sources)

        for layer in self.layers:
            sources = layer(sources, mask)

        return sources


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(EncoderLayer, self).__init__()

        self.self_attn = Sublayer(MultiHeadedAttention(heads_count, d_model, dropout_prob), d_model)
        self.feed_forward = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, d_model)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = self.feed_forward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )

        # Generator Solution 1
        self.generator = Generator(embedding)

        # Generator Solution 2
        # self.generator = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)
        # self.generator.weight = self.embedding.weight

    def forward(self, x, memory, src_mask, tgt_mask=None, state=None):
        # x: (batch_size, seq_len - 1, d_model)
        # memory: (batch_size, seq_len, d_model)

        x = self.embedding(x)
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
        return generated, state

    def init_decoder_state(self, **args):
        return DecoderState()


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    """
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(DecoderLayer, self).__init__()
        self.self_attn = Sublayer(
            MultiHeadedAttention(heads_count, d_model, dropout_prob, mode='self-attention'), d_model)
        self.src_attn = Sublayer(
            MultiHeadedAttention(heads_count, d_model, dropout_prob, mode='memory-attention'), d_model)
        self.feed_forward = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)

    def forward(self, x, memory, src_mask, tgt_mask, layer_cache=None):
        # print('self attention')
        # print('inputs_mask', inputs_mask)
        x = self.self_attn(x, x, x, tgt_mask, layer_cache)
        # print('memory attention')
        x = self.src_attn(x, memory, memory, src_mask, layer_cache)
        x = self.feed_forward(x)
        return x


class DecoderState:

    def __init__(self):
        self.previous_inputs = torch.tensor([])
        self.layer_caches = defaultdict(lambda: {'self-attention': None, 'memory-attention': None})

    def update_state(self, layer_index, layer_mode, key_projected, value_projected):
        self.layer_caches[layer_index][layer_mode] = {
            'key_projected': key_projected,
            'value_projected': value_projected}

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
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources, inputs):
        # sources : (batch_size, sources_len)
        # inputs : (batch_size, targets_len - 1)
        batch_size, sources_len = sources.size()
        batch_size, inputs_len = inputs.size()

        sources_mask = pad_masking(sources, sources_len)
        memory_mask = pad_masking(sources, inputs_len)
        inputs_mask = subsequent_masking(inputs) | pad_masking(inputs, inputs_len)

        # (batch_size, seq_len, d_model)
        memory = self.encoder(sources, sources_mask)  # Context Vectors

        # (batch_size, seq_len, d_model)
        outputs, state = self.decoder(inputs, memory, memory_mask, inputs_mask)
        return outputs


def build_model(config, source_vocabulary_size, target_vocabulary_size):
    if config['positional_encoding']:
        source_embedding = PositionalEncoding(
            num_embeddings=source_vocabulary_size,
            embedding_dim=config['d_model'],
            dim=config['d_model'])  # why dim?
        target_embedding = PositionalEncoding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=config['d_model'],
            dim=config['d_model'])  # why dim?
    else:
        source_embedding = nn.Embedding(
            num_embeddings=source_vocabulary_size,
            embedding_dim=config['d_model'])
        target_embedding = nn.Embedding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=config['d_model'])

    encoder = Encoder(layers_count=config['layers_count'],
                      d_model=config['d_model'],
                      heads_count=config['heads_count'],
                      d_ff=config['d_ff'],
                      dropout_prob=config['dropout_prob'],
                      embedding=source_embedding)

    decoder = Decoder(layers_count=config['layers_count'],
                      d_model=config['d_model'],
                      heads_count=config['heads_count'],
                      d_ff=config['d_ff'],
                      dropout_prob=config['dropout_prob'],
                      embedding=target_embedding)

    model = Transformer(encoder, decoder)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
