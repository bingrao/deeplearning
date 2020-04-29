import torch
from torch.autograd import Variable
from nmt.model.transformer.model import build_model
from benchmarks.example.datasets import IndexedInputTargetTranslationDataset, IndexDictionary
from benchmarks.beam import Beam
from nmt.utils.context import Context
from nmt.utils.pad import subsequent_mask, pad_masking, subsequent_masking



def make_std_mask(tgt, pad):
    """Create a mask to hide padding and future words."""
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


class Predictor:
    def __init__(self, ctx, m, src_dictionary, tgt_dictionary, max_length=30, beam_size=8):
        self.context = ctx
        self.logger = ctx.logger
        self.model = m
        self.source_dictionary = src_dictionary
        self.target_dictionary = tgt_dictionary

        # Get a list index number of a given input string based on source dictionary.
        # Input: a input string
        # output: a corresponding index list
        self.preprocess = IndexedInputTargetTranslationDataset.preprocess(self.source_dictionary)

        # Get a output string by converting a list of index based
        # on target dictionary to a corresponding output string
        # Input: an index list
        # output: a corresponding output string
        self.postprocess = lambda x: ' '.join(
            [token for token in self.target_dictionary.tokenify_indexes(x) if token != '<EndSent>'])

        self.max_length = max_length
        self.beam_size = beam_size
        self.attentions = None
        self.hypothesises = None

        # Set up evaluate model
        self.model.eval()
        # Loading model paramters from previous traning.
        if self.context.project_checkpoint is not None:
            self.checkpoint_filepath = self.context.project_checkpoint
            self.model.load_state_dict(torch.load(self.checkpoint_filepath, map_location='cpu'))
        else:
            self.context.logger.error("[%s] There is no module paramters input and please train it",
                                      self.__class__.__name__,)
            exit(-1)


    def predict_one(self, source=None, num_candidates=None):
        source = self.context.source if source is None else None
        num_candidates = self.context.num_candidates if num_candidates is None else None
        self.logger.debug("[%s] Predict Input Source: %s, nums of Candidates %d", self.__class__.__name__,
                          str(source), num_candidates)

        source_preprocessed = self.preprocess(source)
        self.logger.debug("[%s] The corresponding indexes of [%s]: %s", self.__class__.__name__,
                          str(source), str(source_preprocessed))

        source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?
        length_tensor = torch.tensor(len(source_preprocessed)).unsqueeze(0)
        self.logger.debug("[%s] The index source Tensor: %s, lenght %s", self.__class__.__name__,
                          source_tensor, length_tensor)

        sources_mask = pad_masking(source_tensor, source_tensor.size(1))
        memory = self.model.encode(source_tensor, sources_mask)

        self.logger.debug("[%s] Encoder Source %s, Output %s dimensions", self.__class__.__name__,
                          source_tensor.size(), memory.size())

        memory_mask = sources_mask

        # Repeat beam_size times
        # (beam_size, seq_len, hidden_size)
        memory_beam = memory.detach().repeat(self.beam_size, 1, 1)

        self.logger.debug("[%s] Memory %s dimension", self.__class__.__name__, memory_beam.shape())

        beam = Beam(ctx=self.context,
                    beam_size=self.beam_size,
                    min_length=0,
                    n_top=num_candidates,
                    ranker=None)

        for _ in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            # new_mask = subsequent_masking(new_inputs)
            new_mask = subsequent_masking(new_inputs) | pad_masking(new_inputs, new_inputs.size(1))

            decoder_outputs = self.model.decode(tgt=new_inputs,
                                                memory=memory_beam,
                                                memory_mask=memory_mask,
                                                tgt_mask=new_mask)

            self.logger.debug("[%s] Decoder Input %s, output %s dimensions",
                             self.__class__.__name__, new_inputs.size(), decoder_outputs.size())

            attention = self.model.decoder.layers[-1].src_attn.attention
            self.logger.debug("[%s] attention %s dimension", attention.size())

            beam.advance(decoder_outputs.squeeze(1), attention)
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

    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(context.proj_processed_dir, mode='source',
                                             vocabulary_size=context.vocabulary_size)
    target_dictionary = IndexDictionary.load(context.proj_processed_dir, mode='target',
                                             vocabulary_size=context.vocabulary_size)

    logger.info('Building model...')
    model = build_model(context, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    logger.info("Building Predictor ....")
    predictor = Predictor(ctx=context,
                          m=model,
                          src_dictionary=source_dictionary,
                          tgt_dictionary=target_dictionary)

    logger.info("Get Predict Result ...")
    for index, candidate in enumerate(predictor.predict_one()):
        logger.info(f'Candidate {index} : {candidate}')
