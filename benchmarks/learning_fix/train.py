import torch.nn as nn
from torch.autograd import Variable
import torch
from nmt.data.batch import Batch
from nmt.model.transformer.model import build_model
from nmt.utils.context import Context
from benchmarks.learning_fix.preprocess import dataset
from torch.utils.data import DataLoader
import time

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
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)

    def __call__(self, x, y, norm):

        batch_size, seq_len, vocabulary_size = x.size()

        outputs_flat = x.view(batch_size * seq_len, vocabulary_size)
        targets_flat = y.view(batch_size * seq_len)

        loss = self.base_loss_function(outputs_flat, targets_flat)
        # loss.backward()
        # if self.opt is not None:
        #     self.opt.step()
        #     self.opt.optimizer.zero_grad()
		#
        # x = self.generator(x)
        # loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
        #                       y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # https://github.com/pytorch/pytorch/issues/15585
        # return loss.data[0] * norm
        return loss.data.item() * norm


# class SimpleLossCompute:
#     """A simple loss compute and train function."""
#
#     def __init__(self, generator, criterion, devices=None, opt=None):
#         self.generator = generator
#         self.criterion = criterion
#         self.opt = opt
#         self.pad_index = 0
#         self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
#
#
# def __call__(self, outputs, targets, norm=None):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        batch_loss = self.base_loss_function(outputs_flat, targets_flat)
        batch_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        count = (targets != self.pad_index).sum().item()

        return batch_loss


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
    # print(f"The combine dim is {out_dims}, out {out_tensor.size()}, out_mask {mask.size()}")
    return out_tensor, mask


def custom_collate_fn(batches):
    # print(f"The current batch size: {len(batches)}")
    # for batch in batches:
    #     print(f"src size {batch.src.size()}, \t tgt size {batch.tgt.size()}")
    min_batch_src = [batch.src for batch in batches]
    min_batch_tgt = [batch.tgt for batch in batches]
    src, src_mask = padding_tensor(min_batch_src)
    tgt, tgt_mask = padding_tensor(min_batch_tgt)
    return Batch(src, tgt)



class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, pad_index=0):
        super(TokenCrossEntropyLoss, self).__init__()

        self.pad_index = pad_index
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_index)

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        batch_loss = self.base_loss_function(outputs_flat, targets_flat)

        count = (targets != self.pad_index).sum().item()

        return batch_loss, count


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, ctx, label_smoothing, vocabulary_size, pad_index=0):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.context = ctx
        self.generator = nn.Linear(self.context.d_model, vocabulary_size)
        self.pad_index = pad_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

        smoothing_value = label_smoothing / (vocabulary_size - 2)  # exclude pad and true label
        smoothed_targets = torch.full((vocabulary_size,), smoothing_value)
        smoothed_targets[self.pad_index] = 0
        self.register_buffer('smoothed_targets', smoothed_targets.unsqueeze(0))  # (1, vocabulary_size)

        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets):
        """
        outputs (FloatTensor): (batch_size, seq_len, vocabulary_size)
        targets (LongTensor): (batch_size, seq_len)
        """
        outputs = self.generator(outputs)
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_log_softmax = self.log_softmax(outputs)
        outputs_flat = outputs_log_softmax.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)

        smoothed_targets = self.smoothed_targets.repeat(targets_flat.size(0), 1)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)
        # smoothed_targets: (batch_size * seq_len, vocabulary_size)

        smoothed_targets.masked_fill_((targets_flat == self.pad_index).unsqueeze(1), 0)
        # masked_targets: (batch_size * seq_len, vocabulary_size)

        loss = self.criterion(outputs_flat, smoothed_targets)
        count = (targets != self.pad_index).sum().item()

        return loss, count


def run_epoch(data_iter, model, loss_compute, ctx, task="train"):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        src = batch.src.to(ctx.device)
        tgt = batch.trg.to(ctx.device)
        trg_y = batch.trg_y.to(ctx.device)
        src_mask = batch.src_mask.to(ctx.device)
        tgt_mask = batch.trg_mask.to(ctx.device)
        out = model(src, tgt, src_mask, tgt_mask)
        loss = loss_compute(out, trg_y, batch.ntokens)
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

if __name__ == "__main__":
    # Train the simple copy task.
    ctx = Context(desc="Learning-fix based on Transformer")
    logger = ctx.logger
    nums_batch = ctx.batch_size
    epochs = ctx.epochs
    pad_idx = 0
    logger.info(f"Preparing dataset with batch size ... ")
    train_dataset = dataset(ctx=ctx, target="train",
                            dataset="small", padding=pad_idx, src_vocab=None, tgt_vocab=None)

    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab
    src_vocab_size = len(src_vocab)
    dst_vocab_size = len(tgt_vocab)

    eval_dataset = dataset(ctx=ctx, target="eval",
                           dataset="small", padding=pad_idx, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_dataset = dataset(ctx=ctx, target="test",
                           dataset="small", padding=pad_idx, src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    # GPUs to use
    devices = [0]  # [0, 1, 2, 3]

    logger.info("Build Model ...")
    model = build_model(ctx, src_vocab_size, dst_vocab_size)
    model.cuda() if ctx.is_cuda else None

    # Print out log info for debug ...
    logger.info(model)

    criterion = LabelSmoothing(size=dst_vocab_size, padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda() if ctx.is_cuda else None

    logger.info("Generating Training and Validating Batch datasets ...")
    logger.info(f"Trainning Dataset: epoch[{epochs}], batch size [{nums_batch}]")
    # train_iter = MyIterator(train_dataset, batch_size=nums_batch, device=ctx.device,
    # 						repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    # 						batch_size_fn=batch_size_fn, train=True)

    train_iter = DataLoader(train_dataset, batch_size=nums_batch, shuffle=True, collate_fn=custom_collate_fn)


    # eval_iter = MyIterator(eval_dataset, batch_size=nums_batch, device=ctx.device,
    # 						repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    # 						batch_size_fn=batch_size_fn, train=False)
    eval_iter = DataLoader(eval_dataset, batch_size=nums_batch, shuffle=True, collate_fn=custom_collate_fn)

    # test_iter = MyIterator(test_dataset, batch_size=nums_batch, device=ctx.device,
    # 						repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    # 						batch_size_fn=batch_size_fn, train=False)
    # test_iter = DataLoader(test_dataset, batch_size=nums_batch, shuffle=True, collate_fn=custom_collate_fn)

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
        model_opt = NoamOpt(model_size=model.src_embed[0].d_model,
                            factor=1,
                            warmup=2000,
                            optimizer=torch.optim.Adam(model.parameters(),
                                                       lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(epochs):
            # Set model in train
            model_parallel.train()
            run_epoch(train_iter,
                      model_parallel,
                      loss_func(model.generator, criterion, devices, opt=model_opt),
                      ctx)

            # Evaluation Model
            model_parallel.eval()

            # Get loss
            loss = run_epoch(eval_iter,
                             model_parallel,
                             loss_func(model.generator, criterion, devices, opt=None),
                             ctx)
            logger.info("The loss is %d", loss)
    else:
        model = torch.load("iwslt.pt")

    logger.info("Training is over ...")