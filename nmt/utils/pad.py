import torch
import numpy as np
from torch.autograd import Variable
PAD_TOKEN_INDEX = 0


def pad_masking(x, target_len):
    # x: (batch_size, seq_len)
    batch_size, seq_len = x.size()
    padded_positions = x == PAD_TOKEN_INDEX  # (batch_size, seq_len)
    pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
    return pad_mask


def subsequent_masking(x):
    # x: (batch_size, seq_len - 1)
    batch_size, seq_len = x.size()
    subs_mask = np.triu(np.ones(shape=(seq_len, seq_len)), k=1).astype('uint8')
    subs_mask = torch.tensor(subs_mask).to(x.device)
    subs_mask = subs_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    return subs_mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subs_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subs_mask) == 0