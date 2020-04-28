from nmt.model.common import *


class Decoder(nn.Module):
    def __init__(self, ctx, layer, N, d_model, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.context = ctx
        self.d_model = d_model
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        # Generator Solution 1
        self.generator = Generator(d_model, tgt_vocab_size)

        # Generator Solution 2
        # self.generator = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)
        # self.generator.weight = self.embedding.weight

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: (batch_size, seq_len - 1, d_model)
        # memory: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    """
    def __init__(self, ctx, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.context = ctx
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)