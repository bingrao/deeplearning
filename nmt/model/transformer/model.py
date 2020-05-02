from torch import nn
import copy
from nmt.data.embeddings import Embeddings, PositionalEncoding
from nmt.model.transformer.encoder import Encoder, EncoderLayer
from nmt.model.transformer.decoder import Decoder, DecoderLayer
from nmt.model.common import PositionwiseFeedForward, Generator
from nmt.model.attention import MultiHeadedAttention, MultiHeadAttentionWithMetrics


class Transformer(nn.Module):
    def __init__(self, ctx, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.context = ctx
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        """
        :param src: a batch of input sentense with 2D dimension (batch_size, seq_len)
        :param src_mask:
        :return: a batch of input sentense with encoding embedding, 3D dimentsion (batch_size, seq_len, d_model)
        """
        src_embed = self.src_embed(src)     # after src_embed, return src_embed:(batch_size, seq_len, d_model)
        output = self.encoder(src_embed, src_mask)
        # [Transformer-Encode] The Source torch.Size([10, 35]),
        #                      src_embed torch.Size([10, 35, 128]),
        #                      output torch.Size([10, 35, 128]) dimension
        self.context.logger.debug("[%s-Encode] The Source %s, src_embed %s, output %s dimension",
                                  self.__class__.__name__, src.size(), src_embed.size(), output.size())
        return output

    def decode(self, tgt, memory, memory_mask, tgt_mask):
        """
        :param tgt: The original input of decoder (Shift right one position) with 2D dimension (batch_size, seq_len)
        :param memory: The output of encoder phase (batch_size, seq_len, d_model)
        :param memory_mask:
        :param tgt_mask: The output of decoder phase (batch_size, seq_len, d_model)
        :return:
        """
        tgt_embed = self.tgt_embed(tgt)
        output = self.decoder(tgt_embed, memory, memory_mask, tgt_mask)
        # [Transformer-Decode] The tgt torch.Size([10, 34]),
        #                          tgt_embed torch.Size([10, 34, 128]),
        #                          memory torch.Size([10, 33, 128]),
        #                          output torch.Size([10, 34, 128]) dimension
        self.context.logger.debug("[%s-Decode] The tgt %s, tgt_embed %s, memory %s, output %s dimension",
                                  self.__class__.__name__, tgt.size(), tgt_embed.size(), memory.size(), output.size())
        return output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """

        :param src: The original input to encoder phase with 2D dimension (batch_size, seq_len)
        :param tgt: The original input of decoder (Shift right one position) with 2D dimension (batch_size, seq_len)
        :param src_mask:
        :param tgt_mask:
        :return: The output of transformer with 3D (batch_size, seq_len, d_model)
        """
        # Create mask for src tgt and memory if not provided by user
        from nmt.utils.pad import pad_masking, subsequent_masking
        batch_size, sources_len = src.size()
        batch_size, inputs_len = tgt.size()
        if src_mask is None:
            src_mask = pad_masking(src, sources_len)
            memory_mask = pad_masking(src, inputs_len)
        else:
            memory_mask = src_mask
        if tgt_mask is None:
            tgt_mask = subsequent_masking(tgt) | pad_masking(tgt, inputs_len)

        # Get encoder output, (batch_size, seq_len, d_model)
        memory = self.encode(src, src_mask)  # Context Vectors

        # Get decoder output, (batch_size, seq_len, d_model)
        outputs = self.decode(tgt, memory, memory_mask, tgt_mask)
        return outputs


def build_model(ctx, src_vocab_size, tgt_vocab_size):
    """
    Helper: Construct a model from hyperparameters.
    """
    c = copy.deepcopy
    N = ctx.layers_count  # N=6,
    d_model = ctx.d_model  # d_model=512,
    d_ff = ctx.d_ff  # d_ff=2048,
    h = ctx.heads_count  # h=8,
    dropout = ctx.dropout  # dropout=0.1
    # attn = MultiHeadAttentionWithMetrics(ctx, h, d_model, dropout)
    attn = MultiHeadedAttention(ctx, h, d_model)
    ff = PositionwiseFeedForward(ctx, d_model, d_ff, dropout)
    position = PositionalEncoding(ctx, d_model, dropout)

    model = Transformer(ctx=ctx,
                        encoder=Encoder(ctx,
                                        EncoderLayer(ctx, d_model, c(attn), c(ff), dropout),  # encode layer
                                        N,  # nums of layers in encoder
                                        d_model,  # Dim of vector
                                        src_vocab_size),
                        decoder=Decoder(ctx,
                                        DecoderLayer(ctx, d_model, c(attn), c(attn), c(ff), dropout),  # decode layer
                                        N,  # nums of layers in decoder
                                        d_model,  # Dim of vector
                                        tgt_vocab_size),
                        src_embed=nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
                        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
                        generator=Generator(d_model, tgt_vocab_size))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
