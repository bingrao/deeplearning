# from evaluator import Evaluator
from predict import Predictor
from models import build_model
from datasets import TranslationDataset
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary

from argparse import ArgumentParser
import json
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from tqdm import tqdm
from argument import get_config


class Evaluator:
    def __init__(self, predictor, save_filepath):

        self.predictor = predictor
        self.save_filepath = save_filepath

    def evaluate_dataset(self, test_dataset):
        tokenize = lambda x: x.split()

        predictions = []
        for source, target in tqdm(test_dataset):
            prediction = self.predictor.predict_one(source, num_candidates=1)[0]
            predictions.append(prediction)

        hypotheses = [tokenize(prediction) for prediction in predictions]
        list_of_references = [[tokenize(target)] for source, target in test_dataset]
        smoothing_function = SmoothingFunction()

        with open(self.save_filepath, 'w') as file:
            for (source, target), prediction, hypothesis, references in zip(test_dataset, predictions,
                                                                            hypotheses, list_of_references):
                sentence_bleu_score = sentence_bleu(references, hypothesis,
                                                    smoothing_function=smoothing_function.method3)
                line = "{bleu_score}\t{source}\t{target}\t|\t{prediction}".format(
                    bleu_score=sentence_bleu_score,
                    source=source,
                    target=target,
                    prediction=prediction
                )
                file.write(line + '\n')

        bleu_score = corpus_bleu(list_of_references, hypotheses, smoothing_function=smoothing_function.method3)

        return bleu_score


if __name__ == "__main__":
    config = get_config('Predict translation')

    print('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(config['data_dir'], mode='source', vocabulary_size=config['vocabulary_size'])
    target_dictionary = IndexDictionary.load(config['data_dir'], mode='target', vocabulary_size=config['vocabulary_size'])

    print('Building model...')
    model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    predictor = Predictor(
        preprocess=IndexedInputTargetTranslationDataset.preprocess(source_dictionary),
        postprocess=lambda x: ' '.join([token for token in target_dictionary.tokenify_indexes(x) if token != '<EndSent>']),
        model=model,
        checkpoint_filepath=config["checkpoint"]
    )

    timestamp = datetime.now()
    if config["save_result"] is None:
        eval_filepath = 'logs/eval-{config}-time={timestamp}.csv'.format(
            config=config["config"].replace('/', '-'),
            timestamp=timestamp.strftime("%Y_%m_%d_%H_%M_%S"))
    else:
        eval_filepath = config["save_result"]

    evaluator = Evaluator(
        predictor=predictor,
        save_filepath=eval_filepath
    )

    print('Evaluating...')
    test_dataset = TranslationDataset(config, config["phase"], limit=1000)
    bleu_score = evaluator.evaluate_dataset(test_dataset)
    print('Evaluation time :', datetime.now() - timestamp)

    print("BLEU score :", bleu_score)