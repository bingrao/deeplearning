import torch
from torch import nn
import math
from torch.autograd import Variable


class PositionalEncodingDebug(torch.nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout_prob (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, num_embeddings, embedding_dim, dim, dropout_prob=0., padding_idx=0, max_len=5000):
        super(PositionalEncodingDebug, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embbedding.weight
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        x = self.embbedding(x)
        x = x * math.sqrt(self.dim)
        if step is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


# Position of input source/target word embedding
class PositionalEncoding(torch.nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       d_model (int): embedding size  d_model
    """

    def __init__(self, ctx, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.context = ctx
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # In a sentense, it consists of several words which is indexed from 0.
        # Here max_len means the max number of words can hold by a input sentense.
        # We create refer table [[pe]] with 3D dimension (1 * max_len * d_model),
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)).float())
        # In index of numpy or tensor, start:end:step  0:d_model:2 = 0:-1:2
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x, step=None):
        """
        :param x: 3D deminsion input with a batch of input (batch_size, max_len_centense, d_model).
                  We can see the input like: there are [[batch_size]] size of sentence,
                  In each sentence, there are [[max_len_centense]] size of words
                  each word is embedded as 1D [[d_model]] dimension feature vector.
        :param step:
        :return:
        """

        if step is None:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        else:
            x = x + Variable(self.pe[:, step])
        return self.dropout(x)


# Source and target input embedding
class Embeddings(torch.nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = torch.nn.Embedding(num_embeddings = vocab,
                                      embedding_dim = d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x: The input 2D dimension (batch_size, seq_len)
        :return: The output with 3D (batch_size, seq_len, d_model)
        """
        return self.lut(x) * math.sqrt(self.d_model)

