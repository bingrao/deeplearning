from nmt.model.transformer.model import build_model
from nmt.utils.context import Context
from nmt.data.batch import Batch
from nmt.utils.pad import subsequent_mask
from torchtext import data
from torch.autograd import Variable
from torch import nn
import torch
import time


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    """Defines an iterator that loads batches of data from a Dataset.

       Attributes:
           dataset: The Dataset object to load Examples from.
           batch_size: Batch size.
           batch_size_f: Function of three arguments (new example to add, current
               count of examples in the batch, and current effective batch size)
               that returns the new effective batch size resulting from adding
               that example to a batch. This is useful for dynamic batching, where
               this function would add to the current effective batch size the
               number of tokens in the new example.
           sort_key: A key to use for sorting examples in order to batch together
               examples with similar lengths and minimize padding. The sort_key
               provided to the Iterator constructor overrides the sort_key
               attribute of the Dataset, or defers to it if None.
           train: Whether the iterator represents a train set.
           repeat: Whether to repeat the iterator for multiple epochs. Default: False.
           shuffle: Whether to shuffle examples between epochs.
           sort: Whether to sort examples according to self.sort_key.
               Note that shuffle and sort default to train and (not train).
           sort_within_batch: Whether to sort (in descending order according to
               self.sort_key) within each batch. If None, defaults to self.sort.
               If self.sort is True and this is False, the batch is left in the
               original (ascending) sorted order.
           device (str or `torch.device`): A string or instance of `torch.device`
               specifying which device the Variables are going to be created on.
               If left as default, the tensors will be created on cpu. Default: None.
       """

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

class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.context.d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion, devices=None, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # https://github.com/pytorch/pytorch/issues/15585
        # return loss.data[0] * norm
        return loss.data.item() * norm

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
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions  requires_grad=self.opt is not None)]
            out_column = [[Variable(o[:, i:i + chunk_size].data, requires_grad=True)] for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum().item() / normalize
            total += l.data.item()

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

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for idx in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def rebatch(pad_idx, batch):
    """Fix order in torchtext to match ours"""
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

def run_epoch(data_iter, model, loss_compute, ctx):
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

def run():
    ctx = Context("Train_MultiGPU")
    logger = ctx.logger
    nums_batch = ctx.batch_size
    epochs = ctx.epochs

    # For data loading.
    from torchtext import data, datasets
    logger.info(f"Preparing dataset with batch size ... ")
    import spacy

    # !pip install torchtext spacy
    # !python -m spacy download en
    # !python -m spacy download de

    logger.info("Load en/de data from local ...")
    spacy_de = spacy.load('de', path=ctx.project_raw_dir)
    spacy_en = spacy.load('en', path=ctx.project_raw_dir)


    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    # tokenize_en("I am a Chinese")  --> ['I', 'am', 'a', 'Chinese']
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # Preparing dataset
    logger.info("Build SRC and TGT Fields ...")
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)
    logger.info("Split datasets into train, val and test using SRC/TGT fileds ...")
    MAX_LEN = 150
    # Spilt dataset in root path into train, val, and test dataset
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'),  # A tuple containing the extension to path for each language.
        fields=(SRC, TGT),  # A tuple containing the fields that will be used for data in each language.
        root=ctx.project_raw_dir,  # Root dataset storage directory.
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

    logger.info("Build vocabularies for src and tgt ...")
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    # GPUs to use
    devices = ctx.device_id   #  [0, 1, 2, 3]
    pad_idx = TGT.vocab.stoi["<blank>"]
    logger.info("Build Model ...")
    model = build_model(ctx, len(SRC.vocab), len(TGT.vocab))
    model.cuda() if ctx.is_cuda else None

    # Print out log info for debug ...
    logger.info(model)


    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda() if ctx.is_cuda else None

    logger.info("Generating Training and Validating Batch datasets ...")
    train_iter = MyIterator(train, batch_size=nums_batch, device=ctx.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    logger.info(f"Trainning Dataset: epoch[{epochs}], iterations[{train_iter.iterations}], batch size [{nums_batch}]")

    valid_iter = MyIterator(val, batch_size=nums_batch, device=ctx.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    logger.info(f"Validate Dataset: epoch[{epochs}], iterations[{valid_iter.iterations}], batch size [{nums_batch}]")

    if ctx.is_gpu_parallel:
        # Using multiple GPU resource to train ...
        model_parallel = nn.DataParallel(model, device_ids=devices)
        loss_func = MultiGPULossCompute
    elif ctx.is_cuda:
        # Using Single GPU resource to train ...
        model_parallel = model
        loss_func = SimpleLossCompute
    else:
        # Using Single CPU resource to train ...
        model_parallel = model
        loss_func = SimpleLossCompute

    logger.info("Training Process is begining ...")

    # Training or load model from checkpoint
    if True:
        model_opt = NoamOpt(model_size = model.src_embed[0].d_model,
                            factor = 1,
                            warmup = 2000,
                            optimizer = torch.optim.Adam(model.parameters(),
                                                         lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(epochs):
            # Set model in train
            model_parallel.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_parallel,
                      loss_func(model.generator, criterion, devices, opt=model_opt),
                      ctx)

            # Evaluation Model
            model_parallel.eval()

            # Get loss
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_parallel,
                             loss_func(model.generator, criterion, devices, opt=None),
                             ctx)
            logger.info("The loss is %d", loss)
    else:
        model = torch.load("iwslt.pt")

    logger.info("Training is over and Evaluate Model  ...")
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break

if __name__ == "__main__":
    run()