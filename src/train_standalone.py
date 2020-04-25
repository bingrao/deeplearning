from train import TransformerTrainer
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
from datetime import datetime
import numpy as np
import torch
from models import build_model


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


if __name__ == '__main__':
    run_trainer_standalone(get_config(logger=get_logger()))