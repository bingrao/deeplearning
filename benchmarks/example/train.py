from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from os.path import join
from datetime import datetime
import os
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from benchmarks.example.datasets import IndexedInputTargetTranslationDataset
from benchmarks.example.dictionaries import IndexDictionary
from nmt.metric.metrics import AccuracyMetric
from nmt.utils.pipe import input_target_collate_fn
from nmt.model.transformer.model import build_model
from nmt.utils.context import Context, create_dir


class NoamOptimizer(Adam):

    def __init__(self, params, d_model, factor=2, warmup_steps=4000, betas=(0.9, 0.98), eps=1e-9):
        # self.optimizer = Adam(params, betas=betas, eps=eps)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr = 0
        self.step_num = 0
        self.factor = factor
        super(NoamOptimizer, self).__init__(params, betas=betas, eps=eps)

    def step(self, closure=None):
        self.step_num += 1
        self.lr = self.lrate()
        for group in self.param_groups:
            group['lr'] = self.lr
        super(NoamOptimizer, self).step()

    def lrate(self):
        return self.factor * self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))



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


class TransformerTrainer:
    def __init__(self, model,       # Transformer model
                 train_dataloader,  # train dataset loader
                 val_dataloader,    # validate dataset loader
                 loss_function,     # loss function
                 metric_function,   # Accuracy Function
                 optimizer,         # Model Optimizer
                 # logger,            # logger agent
                 run_name,          # String Name
                 ctx):

        self.logger = ctx.logger
        self.proj_processed_dir = ctx.project_processed_dir
        self.context = ctx

        self.device = ctx.device
        self.run_name = run_name
        self.model = model.to(self.device) if ctx.is_cuda else model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_function = loss_function.to(self.device) if ctx.is_cuda else model
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.clip_grads = ctx.clip_grads

        self.checkpoint = ctx.project_checkpoint
        self.checkpoint_dir = join(os.path.dirname(ctx.project_raw_dir),'checkpoints', run_name)
        create_dir(self.checkpoint_dir)

        self.print_every = ctx.print_every
        self.save_every = ctx.save_every

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

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
            "Learning rate: {current_lr:<.4} ")

    def run_epoch(self, dataloader, mode='train'):
        batch_losses = []
        batch_counts = []
        batch_metrics = []
        for sources, inputs, targets in tqdm(dataloader):
            sources = sources.to(self.device) if self.context.is_cuda else sources
            inputs = inputs.to(self.device) if self.context.is_cuda else inputs
            targets = targets.to(self.device) if self.context.is_cuda else targets
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

    def run(self, epochs=100):
        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch
            self.logger.debug("Train self epoch [%s], epochs [%s]", self.epoch, epochs)
            self.model.train()
            epoch_start_time = datetime.now()
            train_epoch_loss, train_epoch_metrics = self.run_epoch(self.train_dataloader, mode='train')
            epoch_end_time = datetime.now()

            self.model.eval()

            val_epoch_loss, val_epoch_metrics = self.run_epoch(self.val_dataloader, mode='val')

            if (epoch % self.print_every == 0 or epoch == epochs) and self.logger:
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

            if epoch % self.save_every == 0 or epoch == epochs:
                self._save_model(epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics)

    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics):

        checkpoint_filename = self.save_format.format(
            epoch=epoch,
            val_loss=val_epoch_loss,
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        if self.checkpoint is None:
            checkpoint_filepath = join(self.checkpoint_dir, checkpoint_filename)
        else:
            checkpoint_filepath = self.checkpoint

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


def run_trainer_standalone(ctx):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    logger = ctx.logger

    run_name = (
        "d_model={d_model}-"
        "layers_count={layers_count}-"
        "heads_count={heads_count}-"
        "pe={positional_encoding}-"
        "optimizer={optimizer}-"
        "{timestamp}"
    ).format(**ctx.config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger.info(f'Run name : {run_name}')
    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(ctx.project_processed_dir, mode='source',
                                             vocabulary_size=ctx.vocabulary_size)
    logger.info(f'Source dictionary vocabulary Size: {source_dictionary.vocabulary_size} tokens')

    target_dictionary = IndexDictionary.load(ctx.project_processed_dir, mode='target',
                                             vocabulary_size=ctx.vocabulary_size)
    logger.info(f'Target dictionary vocabulary Size: {target_dictionary.vocabulary_size} tokens')

    logger.info('Building model...')
    model = build_model(ctx, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    logger.info(model)
    logger.info('Encoder : {parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.encoder.parameters()])))
    logger.info('Decoder : {parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.decoder.parameters()])))
    logger.info('Total : {parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    logger.info('Loading datasets...')
    train_dataset = IndexedInputTargetTranslationDataset(ctx=ctx, phase='train')

    val_dataset = IndexedInputTargetTranslationDataset(ctx=ctx, phase='val')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=ctx.batch_size,
        shuffle=True,
        collate_fn=input_target_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=ctx.batch_size,
        collate_fn=input_target_collate_fn)

    if ctx.label_smoothing > 0.0:
        loss_function = LabelSmoothingLoss(ctx=ctx, label_smoothing=ctx.label_smoothing,
                                           vocabulary_size=target_dictionary.vocabulary_size)
    else:
        loss_function = TokenCrossEntropyLoss()

    accuracy_function = AccuracyMetric()

    if ctx.optimizer == 'Noam':
        optimizer = NoamOptimizer(model.parameters(), d_model=ctx.d_model)
    elif ctx.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=ctx.lr)
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
        run_name=run_name,
        ctx=ctx
    )

    trainer.run(ctx.epochs)

    return trainer


if __name__ == '__main__':
    run_trainer_standalone(Context(desc="Train Example Project with GPU Resource!"))