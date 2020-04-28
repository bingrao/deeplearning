from torch import nn
import copy
from nmt.data.embeddings import Embeddings, PositionalEncoding
from nmt.model.transformer.encoder import Encoder, EncoderLayer
from nmt.model.transformer.decoder import Decoder, DecoderLayer
from nmt.model.common import MultiHeadedAttention, PositionwiseFeedForward, Generator, MultiHeadAttention


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
        src_embed = self.src_embed(src)
        output = self.encoder(src_embed, src_mask)
        self.context.logger.debug("[%s-Encode] The Source, src_embed, output dimension: %s, %s, %s",
                                  self.__class__.__name__,
                                  src.size(),
                                  src_embed.size(),
                                  output.size())
        return output

    def decode(self, tgt, memory, memory_mask, tgt_mask):
        tgt_embed = self.tgt_embed(tgt)
        output = self.decoder(tgt_embed, memory, memory_mask, tgt_mask)
        self.context.logger.debug("[%s-Decode] The tgt, tgt_embed, memory, output dimension: %s, %s, %s, %s",
                                  self.__class__.__name__,
                                  tgt.size(),
                                  tgt_embed.size(),
                                  memory.size(),
                                  output.size())
        return output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # batch_size, sources_len = src.size()
        # batch_size, inputs_len = tgt.size()

        """
        RuntimeError: CUDA out of memory. Tried to allocate 68.05 GiB 
        (GPU 0; 15.75 GiB total capacity; 10.36 GiB already allocated; 
        4.35 GiB free; 10.43 GiB reserved in total by PyTorch)
        """
        # if src_mask is None:
        #     src_mask = pad_masking(src, sources_len)
        #     memory_mask = pad_masking(src, inputs_len)
        # else:
        #     memory_mask = src_mask
        # if tgt_mask is None:
        #     tgt_mask = subsequent_masking(tgt) | pad_masking(tgt, inputs_len)

        # Get encoder output, (batch_size, seq_len, d_model)


        memory = self.encode(src, src_mask)  # Context Vectors
        # Get decoder output, (batch_size, seq_len, d_model)
        outputs = self.decode(tgt, memory, src_mask, tgt_mask)
        return outputs


def build_model(ctx, src_vocab_size, tgt_vocab_size):
    """
    Helper: Construct a model from hyperparameters.
    """
    c = copy.deepcopy
    config = ctx.config
    N = config['layers_count']  # N=6,
    d_model = config['d_model']  # d_model=512,
    d_ff = config['d_ff']  # d_ff=2048,
    h = config['heads_count']  # h=8,
    dropout = config['dropout_prob']  # dropout=0.1
    attn = MultiHeadAttention(h, d_model, dropout)
    # attn = MultiHeadedAttention(ctx, h, d_model)
    ff = PositionwiseFeedForward(ctx, d_model, d_ff, dropout)
    position = PositionalEncoding(ctx, d_model, dropout)

    # from nmt.data.embeddings import PositionalEncodingDebug
    # if config['positional_encoding']:
    #     source_embedding = PositionalEncodingDebug(
    #         num_embeddings=src_vocab_size,
    #         embedding_dim=config['d_model'],
    #         dim=config['d_model'])  # why dim?
    #     target_embedding = PositionalEncodingDebug(
    #         num_embeddings=tgt_vocab_size,
    #         embedding_dim=config['d_model'],
    #         dim=config['d_model'])  # why dim?
    # else:
    #     source_embedding = nn.Embedding(
    #         num_embeddings=src_vocab_size,
    #         embedding_dim=config['d_model'])
    #     target_embedding = nn.Embedding(
    #         num_embeddings=tgt_vocab_size,
    #         embedding_dim=config['d_model'])

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
