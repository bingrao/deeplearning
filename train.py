from models import build_model
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary
from losses import TokenCrossEntropyLoss, LabelSmoothingLoss
from metrics import AccuracyMetric
from optimizers import NoamOptimizer
from utils.log import get_logger
from utils.pipe import input_target_collate_fn
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import random
import torch
import numpy as np
from tqdm import tqdm
from os.path import dirname, abspath, join, exists
from os import makedirs
from datetime import datetime
import json
import json
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from models import build_model
from torchtext import data, datasets

PAD_INDEX = 0
BASE_DIR = dirname(abspath(__file__))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

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
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
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




class TransformerTrainer:

    def __init__(self, model,
                 train_dataloader, val_dataloader,
                 loss_function, metric_function, optimizer,
                 logger, run_name,
                 save_config, save_checkpoint,
                 config):

        self.config = config
        self.device = torch.device(self.config['device'])

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_function = loss_function.to(self.device)
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.clip_grads = self.config['clip_grads']

        self.logger = logger
        self.checkpoint_dir = join(BASE_DIR, 'checkpoints', run_name)

        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        if save_config is None:
            config_filepath = join(self.checkpoint_dir, 'config.json')
        else:
            config_filepath = save_config
        with open(config_filepath, 'w') as config_file:
            json.dump(self.config, config_file)

        self.print_every = self.config['print_every']
        self.save_every = self.config['save_every']

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

        self.save_checkpoint = save_checkpoint
        self.save_format = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth'

        self.log_format = (
            "Epoch: {epoch:>3} "
            "Progress: {progress:<.1%} "
            "Elapsed: {elapsed} "
            "Examples/second: {per_second:<.1} "
            "Train Loss: {train_loss:<.6} "
            "Val Loss: {val_loss:<.6} "
            "Train Metrics: {train_metrics} "
            "Val Metrics: {val_metrics} "
            "Learning rate: {current_lr:<.4} "
        )

    def run_epoch(self, dataloader, mode='train'):
        batch_losses = []
        batch_counts = []
        batch_metrics = []
        for sources, inputs, targets in tqdm(dataloader):
            sources, inputs, targets = sources.to(self.device), inputs.to(self.device), targets.to(self.device)
            outputs = self.model(sources, inputs)

            batch_loss, batch_count = self.loss_function(outputs, targets)

            if mode == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            batch_losses.append(batch_loss.item())
            batch_counts.append(batch_count)

            batch_metric, batch_metric_count = self.metric_function(outputs, targets)
            batch_metrics.append(batch_metric)

            assert batch_count == batch_metric_count

            if self.epoch == 0:  # for testing
                return float('inf'), [float('inf')]

        epoch_loss = sum(batch_losses) / sum(batch_counts)
        epoch_accuracy = sum(batch_metrics) / sum(batch_counts)
        epoch_perplexity = float(np.exp(epoch_loss))
        epoch_metrics = [epoch_perplexity, epoch_accuracy]

        return epoch_loss, epoch_metrics

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            self.model.train()

            epoch_start_time = datetime.now()
            train_epoch_loss, train_epoch_metrics = self.run_epoch(self.train_dataloader, mode='train')
            epoch_end_time = datetime.now()

            self.model.eval()

            val_epoch_loss, val_epoch_metrics = self.run_epoch(self.val_dataloader, mode='val')

            if epoch % self.print_every == 0 and self.logger:
                per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                log_message = self.log_format.format(epoch=epoch,
                                                     progress=epoch / epochs,
                                                     per_second=per_second,
                                                     train_loss=train_epoch_loss,
                                                     val_loss=val_epoch_loss,
                                                     train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
                                                     val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
                                                     current_lr=current_lr,
                                                     elapsed=self._elapsed_time()
                                                     )

                self.logger.info(log_message)

            if epoch % self.save_every == 0:
                self._save_model(epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics)

    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics):

        checkpoint_filename = self.save_format.format(
            epoch=epoch,
            val_loss=val_epoch_loss,
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        if self.save_checkpoint is None:
            checkpoint_filepath = join(self.checkpoint_dir, checkpoint_filename)
        else:
            checkpoint_filepath = self.save_checkpoint

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'train_metrics': train_epoch_metrics,
            'val_loss': val_epoch_loss,
            'val_metrics': val_epoch_metrics,
            'checkpoint': checkpoint_filepath,
        }

        if self.epoch > 0:
            torch.save(self.model.state_dict(), checkpoint_filepath)
            self.history.append(save_state)

        representative_val_metric = val_epoch_metrics[0]
        if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_loss_at_best = val_epoch_loss
            self.train_loss_at_best = train_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.val_metrics_at_best = val_epoch_metrics
            self.best_checkpoint_filepath = checkpoint_filepath

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds


def run_trainer_standalone(config):
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

    logger = get_logger(run_name, save_log=config['save_log'])
    logger.info(f'Run name : {run_name}')
    logger.info(config)

    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(config['data_dir'], mode='source', vocabulary_size=config['vocabulary_size'])
    target_dictionary = IndexDictionary.load(config['data_dir'], mode='target', vocabulary_size=config['vocabulary_size'])
    logger.info(f'Source dictionary vocabulary : {source_dictionary.vocabulary_size} tokens')
    logger.info(f'Target dictionary vocabulary : {target_dictionary.vocabulary_size} tokens')

    logger.info('Building model...')
    model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    logger.info(model)
    logger.info('Encoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.encoder.parameters()])))
    logger.info('Decoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.decoder.parameters()])))
    logger.info('Total : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.parameters()])))

    logger.info('Loading datasets...')
    train_dataset = IndexedInputTargetTranslationDataset(
        data_dir=config['data_dir'],
        phase='train',
        vocabulary_size=config['vocabulary_size'],
        limit=config['dataset_limit'])

    val_dataset = IndexedInputTargetTranslationDataset(
        data_dir=config['data_dir'],
        phase='val',
        vocabulary_size=config['vocabulary_size'],
        limit=config['dataset_limit'])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=input_target_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=input_target_collate_fn)

    if config['label_smoothing'] > 0.0:
        loss_function = LabelSmoothingLoss(label_smoothing=config['label_smoothing'],
                                           vocabulary_size=target_dictionary.vocabulary_size)
    else:
        loss_function = TokenCrossEntropyLoss()

    accuracy_function = AccuracyMetric()

    if config['optimizer'] == 'Noam':
        optimizer = NoamOptimizer(model.parameters(), d_model=config['d_model'])
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    else:
        raise NotImplementedError()

    logger.info('Start training...')
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        metric_function=accuracy_function,
        optimizer=optimizer,
        logger=logger,
        run_name=run_name,
        save_config=config['save_config'],
        save_checkpoint=config['save_checkpoint'],
        config=config
    )

    trainer.run(config['epochs'])

    return trainer


def get_config(desc='Train Transformer'):
    parser = ArgumentParser(description='Train Transformer')
    parser.add_argument('--config', type=str, default=None)

    parser.add_argument('--data_dir', type=str, default='data/example/processed')
    parser.add_argument('--save_config', type=str, default=None)
    parser.add_argument('--save_checkpoint', type=str, default=None)
    parser.add_argument('--save_log', type=str, default=None)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu_idx', type=str, default=[0])

    parser.add_argument('--dataset_limit', type=int, default=None)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)

    parser.add_argument('--vocabulary_size', type=int, default=None)
    parser.add_argument('--positional_encoding', action='store_true')

    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--layers_count', type=int, default=6)
    parser.add_argument('--heads_count', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--dropout_prob', type=float, default=0.1)

    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clip_grads', action='store_true')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary

    return config
