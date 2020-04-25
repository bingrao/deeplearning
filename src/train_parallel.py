import torch.nn as nn
import time
from torch.autograd import Variable
from torchtext import data
from argument import get_config
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary
from optimizers import NoamOptimizer
from utils.log import get_logger
from torch.optim import Adam
import random
from datetime import datetime
import numpy as np
import torch
from models import build_model

logger = get_logger("train_example")
global max_src_in_batch, max_tgt_in_batch


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subs_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subs_mask) == 0


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    """Fix order in torchtext to match ours"""
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


# Finally to really target fast training, we will use multi-gpu.
# This code implements multi-gpu word generation. It is not specific to
# transformer so I won’t go into too much detail. The idea is to split up
# word generation at training time into chunks to be processed in parallel
# across many different gpus. We do this using pytorch parallel primitives:

# replicate - split modules onto different gpus.
# scatter - split batches onto different gpus
# parallel_apply - apply module to batches on different gpus
# gather - pull scattered data back onto one gpu.
# nn.DataParallel - a special module wrapper that calls these all before evaluating.

class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=self.devices)
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


##################################################################################
# Now we consider a real-world example using the IWSLT German-English Translation task.
# This task is much smaller than the WMT task considered in the paper, but it illustrates
# the whole system. We also show how to use multi-gpu processing to make it really fast.

def train_with_example_dataset(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_name_format = (
        "d_model={d_model}-"
        "layers_count={layers_count}-"
        "heads_count={heads_count}-"
        "pe={positional_encoding}-"
        "optimizer={optimizer}-"
        "{timestamp}"
    )

    run_name = run_name_format.format(**config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger.info(f'Run name : {run_name}')

    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(config['data_dir'], mode='source',
                                             vocabulary_size=config['vocabulary_size'])
    logger.info(f'Source dictionary vocabulary Size: {source_dictionary.vocabulary_size} tokens')

    target_dictionary = IndexDictionary.load(config['data_dir'], mode='target',
                                             vocabulary_size=config['vocabulary_size'])
    logger.info(f'Target dictionary vocabulary Size: {target_dictionary.vocabulary_size} tokens')

    logger.info('Building model...')
    model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    logger.info(model)
    logger.info('Encoder : {parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.encoder.parameters()])))
    logger.info('Decoder : {parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.decoder.parameters()])))
    logger.info('Total : {parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    logger.info('Loading datasets...')
    train_dataset = IndexedInputTargetTranslationDataset(config=config, phase='train')

    val_dataset = IndexedInputTargetTranslationDataset(config=config, phase='val')

    if config['optimizer'] == 'Noam':
        optimizer = NoamOptimizer(model.parameters(), d_model=config['d_model'])
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    else:
        # raise NotImplementedError()

	pad_idx = 0
    devices = [0, 1, 2, 3]
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()

    train_iter = MyIterator(train_dataset, batch_size=config['batch_size'], device=0,
                            repeat=False, batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val_dataset, batch_size=config['batch_size'], device=0,
                            repeat=False, batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    # Training the System
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=optimizer))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))


if __name__ == "__main__":
    train_with_example_dataset(get_config(logger=logger))