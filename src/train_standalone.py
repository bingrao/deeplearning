from argument import get_config
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary
from losses import TokenCrossEntropyLoss, LabelSmoothingLoss
from metrics import AccuracyMetric
from optimizers import NoamOptimizer
from utils.log import get_logger
from utils.pipe import input_target_collate_fn
from torch.optim import Adam
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from os.path import join, exists
from os import makedirs
from datetime import datetime
import json
import numpy as np
import torch
from models import build_model
from context import Context


class TransformerTrainer:
    def __init__(self, model,       # Transformer model
                 train_dataloader,  # train dataset loader
                 val_dataloader,    # validate dataset loader
                 loss_function,     # loss function
                 metric_function,   # Accuracy Function
                 optimizer,         # Model Optimizer
                 run_name,          # String Name
                 ctx):

        self.context = ctx
        self.config = ctx.config
        self.logger = ctx.logger
        self.device = torch.device(self.config['device'])
        self.save_data_dir = self.config["save_data_dir"]

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_function = loss_function.to(self.device)
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.clip_grads = self.config['clip_grads']


        self.checkpoint_dir = join(self.save_data_dir, 'checkpoints', run_name)

        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        if self.config["save_config"] is None:
            config_filepath = join(self.save_data_dir, 'checkpoints','config.json')
        else:
            config_filepath = self.config["save_config"]

        with open(config_filepath, 'w') as config_file:
            json.dump(self.config, config_file)

        self.print_every = self.config['print_every']
        self.save_every = self.config['save_every']

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

        self.checkpoint = self.config["checkpoint"]
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
            sources, inputs, targets = sources.to(self.device), inputs.to(self.device), targets.to(self.device)
            outputs = self.model.forward(sources, inputs)

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
    config = ctx.config
    logger = ctx.logger

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
    logger.info(config)

    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(config['save_data_dir'], mode='source',
                                             vocabulary_size=config['vocabulary_size'])
    logger.info(f'Source dictionary vocabulary Size: {source_dictionary.vocabulary_size} tokens')

    target_dictionary = IndexDictionary.load(config['save_data_dir'], mode='target',
                                             vocabulary_size=config['vocabulary_size'])
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
        run_name=run_name,
        ctx=ctx
    )

    trainer.run(config['epochs'])

    return trainer


if __name__ == '__main__':
    run_trainer_standalone(Context(desc="train"))