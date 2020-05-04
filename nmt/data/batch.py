from nmt.utils.pad import make_std_mask
import time

class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.tgt = trg
        self.pad = pad
        # -1 means last dimension, -2 last second dimension
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1] # remove last column
            self.trg_y = trg[:, 1:] # remove first column
            self.trg_mask = make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

def custom_collate_fn(batches):
    """
    Combine a list of single batch file into a mini-batch
    :param batches: a list of batch contain src and tgt tensor
    :return: return a new Batch to represent all sub batches
    """
    def padding_tensor(sequences):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        max_len = max([s.size(1) for s in sequences]) + 1
        out_dims = (num, max_len)

        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(1)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return out_tensor, mask
    min_batch_src = [batch.src for batch in batches]
    min_batch_tgt = [batch.tgt for batch in batches]
    src, src_mask = padding_tensor(min_batch_src)
    tgt, tgt_mask = padding_tensor(min_batch_tgt)
    return Batch(src, tgt, batches[0].pad)

def default_run_epoch(data_iter, model, loss_compute, ctx):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            ctx.logger.info("Epoch Step: %d Loss: %f Tokens per Sec: %f",
                            i, loss / batch.ntokens, tokens / elapsed)
            start = time.time()
            tokens = 0
    return total_loss / total_tokens