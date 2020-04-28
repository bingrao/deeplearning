from nmt.model import build_model, make_std_mask
from nmt.data.datasets import IndexedInputTargetTranslationDataset
from nmt.data import IndexDictionary
from nmt.predict import Beam
import torch
from nmt.utils.context import Context


class Predictor:
    def __init__(self, ctx, m, src_dictionary, tgt_dictionary, max_length=30, beam_size=8):
        self.context = ctx
        self.config = ctx.config
        self.logger = ctx.logger
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

        self.logger.info("########Source: %s", str(source))
        source_preprocessed = self.preprocess(source)
        self.logger.info("########source_preprocessed: %s", str(source_preprocessed))

        source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?
        length_tensor = torch.tensor(len(source_preprocessed)).unsqueeze(0)
        self.logger.info("########source_tensor: %s", str(source_tensor))

        # sources_mask = pad_masking(source_tensor, source_tensor.size(1))
        sources_mask = (source_tensor != 0).unsqueeze(-2)

        memory = self.model.encode(source_tensor, sources_mask)
        self.logger.info("#########Encoder Result(Memory): %s", memory)
        memory_mask = sources_mask

        decoder_state = self.model.decoder.init_decoder_state()

        # self.logger.info('decoder_state src: %s', decoder_state.src.shape)
        # self.logger.info('previous_input previous_input: %s', decoder_state.previous_input)
        # self.logger.info('previous_input previous_layer_inputs: %s ', decoder_state.previous_layer_inputs)


        # Repeat beam_size times
        memory_beam = memory.detach().repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, hidden_size)
        self.logger.info("#########Encoder Result(Memory Beam): %s", memory_beam)

        beam = Beam(beam_size=self.beam_size, min_length=0, n_top=num_candidates, ranker=None)

        for _ in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            # new_mask = subsequent_masking(new_inputs)
            new_mask = make_std_mask(new_inputs, 0)
            self.logger.info("#########Encoder Input: %s", new_inputs)
            decoder_outputs, decoder_state = self.model.decode(tgt=new_inputs,
                                                               memory=memory_beam,
                                                               memory_mask=memory_mask,
                                                               tgt_mask=new_mask,
                                                               state=decoder_state)

            self.logger.info("#########Decoder Result: %s", decoder_outputs)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            attention = self.model.decoder.layers[-1].src_attn.attention
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
    logger.info("Building Predictor ....")
    predictor = Predictor(ctx=context,m=model,src_dictionary=source_dictionary, tgt_dictionary=target_dictionary)

    logger.info("Get Predict Result ...")
    for index, candidate in enumerate(predictor.predict_one()):
        logger.info(f'Candidate {index} : {candidate}')
