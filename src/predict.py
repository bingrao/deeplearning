from models import build_model
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary
from beam import Beam
from utils.pad import pad_masking
import torch
from utils.context import Context


class Predictor:
    def __init__(self, ctx, m, src_dictionary, tgt_dictionary, max_length=30, beam_size=8):
        self.context = ctx
        self.config = self.context.config
        self.logger = self.context.logger
        self.model = m
        self.source_dictionary = src_dictionary
        self.target_dictionary = tgt_dictionary

        self.preprocess = IndexedInputTargetTranslationDataset.preprocess(self.source_dictionary)

        self.postprocess = lambda x: ' '.join(
            [token for token in self.target_dictionary.tokenify_indexes(x) if token != '<EndSent>'])

        self.max_length = max_length
        self.beam_size = beam_size
        self.attentions = None
        self.hypothesises = None
        self.model.eval()
        self.checkpoint_filepath = self.config["checkpoint"]
        self.model.load_state_dict(torch.load(self.checkpoint_filepath, map_location='cpu'))


    def predict_one(self, source=None, num_candidates=None):

        if source is None:
            source = self.config["source"]
        if num_candidates is None:
            num_candidates = self.config["num_candidates"]

        # self.logger.info("########Source: ", str(source))
        source_preprocessed = self.preprocess(source)
        # self.logger.info("########source_preprocessed: ", str(source_preprocessed))

        source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?
        length_tensor = torch.tensor(len(source_preprocessed)).unsqueeze(0)
        # self.logger.info("########source_tensor", str(source_tensor))

        sources_mask = pad_masking(source_tensor, source_tensor.size(1))
        memory_mask = pad_masking(source_tensor, 1)
        # self.logger.info("########sources_mask", str(sources_mask))
        # self.logger.info("########memory_mask", str(memory_mask))
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
            decoder_outputs, decoder_state = self.model.decoder(new_inputs,
                                                                memory_beam,
                                                                memory_mask,
                                                                state=decoder_state)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            attention = self.model.decoder.layers[-1].src_attn.sublayer.attention
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


if __name__ == "__main__":

    context = Context(desc="Prediction")
    logger = context.logger
    config = context.config

    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(config['save_data_dir'], mode='source',
                                             vocabulary_size=config['vocabulary_size'])
    target_dictionary = IndexDictionary.load(config['save_data_dir'], mode='target',
                                             vocabulary_size=config['vocabulary_size'])

    logger.info('Building model...')
    model = build_model(context, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    predictor = Predictor(ctx=context,m=model,src_dictionary=source_dictionary, tgt_dictionary=target_dictionary)

    for index, candidate in enumerate(predictor.predict_one(config["source"],
                                                            num_candidates=config["num_candidates"])):
        logger.info(f'Candidate {index} : {candidate}')
