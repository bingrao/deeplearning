
from nmt.model.common import *


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, ctx, layer, N, d_model, src_vocab_size):
        super(Encoder, self).__init__()
        self.context = ctx
        self.d_model = d_model
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        :param x: embedded_sequence, (batch_size, seq_len, embed_size)
        :param mask:
        :return: encoded_sequence, (batch_size, seq_len, embed_size)
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
    def __init__(self, ctx, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.context = ctx
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.dropout = nn.Dropout(dropout)
        self.index = 0

    def forward(self, x, mask):
        """
        :param x: encoded/embedded_sequence, (batch_size, seq_len, d_model)
        :param mask:
        :return: encoded_sequence, (batch_size, seq_len, d_model)
        """
        # x: (batch_size, seq_len, d_model)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # x = self.dropout(x)  # Optional
        return self.sublayer[1](x, self.feed_forward)