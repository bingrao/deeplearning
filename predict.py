# from predictors import Predictor
from models import build_model
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary

from argparse import ArgumentParser
import json

from beam import Beam
from utils.pad import pad_masking
import torch


class Predictor:

    def __init__(self, preprocess, postprocess, model, checkpoint_filepath, max_length=30, beam_size=8):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.model = model
        self.max_length = max_length
        self.beam_size = beam_size

        self.model.eval()
        checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def predict_one(self, source, num_candidates=5):
        source_preprocessed = self.preprocess(source)
        source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?
        length_tensor = torch.tensor(len(source_preprocessed)).unsqueeze(0)

        sources_mask = pad_masking(source_tensor, source_tensor.size(1))
        memory_mask = pad_masking(source_tensor, 1)
        memory = self.model.encoder(source_tensor, sources_mask)

        decoder_state = self.model.decoder.init_decoder_state()
        # print('decoder_state src', decoder_state.src.shape)
        # print('previous_input previous_input', decoder_state.previous_input)
        # print('previous_input previous_layer_inputs ', decoder_state.previous_layer_inputs)


        # Repeat beam_size times
        memory_beam = memory.detach().repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, hidden_size)

        beam = Beam(beam_size=self.beam_size, min_length=0, n_top=num_candidates, ranker=None)

        for _ in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            decoder_outputs, decoder_state = self.model.decoder(new_inputs, memory_beam,
                                                                            memory_mask,
                                                                            state=decoder_state)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            attention = self.model.decoder.decoder_layers[-1].memory_attention_layer.sublayer.attention
            beam.advance(decoder_outputs.squeeze(1), attention)

            beam_current_origin = beam.get_current_origin()  # (beam_size, )
            decoder_state.beam_update(beam_current_origin)

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=num_candidates)
        hypothesises, attentions = [], []
        for i, (times, k) in enumerate(ks[:num_candidates]):
            hypothesis, attention = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
            attentions.append(attention)

        self.attentions = attentions
        self.hypothesises = [[token.item() for token in h] for h in hypothesises]
        hs = [self.postprocess(h) for h in self.hypothesises]
        return list(reversed(hs))

parser = ArgumentParser(description='Predict translation')
parser.add_argument('--source', type=str)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--num_candidates', type=int, default=3)

args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

print('Constructing dictionaries...')
source_dictionary = IndexDictionary.load(config['data_dir'], mode='source', vocabulary_size=config['vocabulary_size'])
target_dictionary = IndexDictionary.load(config['data_dir'], mode='target', vocabulary_size=config['vocabulary_size'])

print('Building model...')
model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

predictor = Predictor(
    preprocess=IndexedInputTargetTranslationDataset.preprocess(source_dictionary),
    postprocess=lambda x: ' '.join([token for token in target_dictionary.tokenify_indexes(x) if token != '<EndSent>']),
    model=model,
    checkpoint_filepath=args.checkpoint
)

for index, candidate in enumerate(predictor.predict_one(args.source, num_candidates=args.num_candidates)):
    print(f'Candidate {index} : {candidate}')
