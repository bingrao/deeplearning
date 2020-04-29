from os.path import join, exists
from os import makedirs
from nmt.data.dictionaries import IndexDictionary
from nmt.utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator
from nmt.data.dictionaries import START_TOKEN, END_TOKEN
import logging
import torch
from nmt.utils.context import Context

UNK_INDEX = 1


def gen_raw_data(ctx, phase):
    assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'"
    # Load Source Raw dataset

    if phase == 'train':
        src_dir = ctx.train_src_dataset
        tgt_dir = ctx.train_dst_dataset
    else:
        src_dir = ctx.val_src_dataset
        tgt_dir = ctx.val_dst_dataset

    with open(src_dir) as source_file:
        source_data = source_file.readlines()
    # Load Target Raw dataset
    with open(tgt_dir) as target_file:
        target_data = target_file.readlines()

    with open(join(ctx.project_processed_dir, f'raw-{phase}.txt'), 'w') as file:
        for source_line, target_line in zip(source_data, target_data):
            source_line = source_line.strip()
            target_line = target_line.strip()
            file.write(f'{source_line}\t{target_line}\n')


class TranslationDataset:
    """
    :return
      - self.data: a list of train/val raw data pairs (src, tgt)
    """
    def __init__(self, ctx, phase, limit=None):
        assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'"
        self.context = ctx
        self.logger = ctx.logger

        self.limit = limit
        self.phase = phase
        self.raw_data_path = join(self.context.project_processed_dir, f'raw-{phase}.txt')
        self.data = []

        # Raw dataset does not exist, and then create
        if not exists(self.raw_data_path):
            gen_raw_data(ctx=self.context, phase=self.phase)

        # Append (src,tgt) to self.data
        with open(join(self.context.project_processed_dir, f'raw-{phase}.txt')) as file:
            for line in file:
                source, target = line.strip().split('\t')
                self.data.append((source, target))

    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()

        return self.data[item]

    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else:
            return self.limit

    @staticmethod
    def prepare(ctx):
        """
        1.  Combine train source and target into a single Raw file
        2.  Combine validate source and target into a single Raw file
        """
        for phase in ('train', 'val'):
            gen_raw_data(ctx=ctx, phase=phase)


class TranslationDatasetOnTheFly:
    """
    :returns
      - self.source_data: a list of train/val raw source data
      - self.target_data: a list of train/val raw target data
    """
    def __init__(self, ctx, phase, limit=None):
        assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'"
        self.context = ctx
        self.limit = limit
        self.logger = ctx.logger

        if phase == 'train':
            src_dir = ctx.train_src_dataset
            tgt_dir = ctx.train_dst_dataset
        else:
            src_dir = ctx.val_src_dataset
            tgt_dir = ctx.val_dst_dataset

        with open(src_dir) as source_file:
            self.source_data = source_file.readlines()

        with open(tgt_dir) as target_file:
            self.target_data = target_file.readlines()

    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()

        source = self.source_data[item].strip()
        target = self.target_data[item].strip()
        return source, target

    def __len__(self):
        if self.limit is None:
            return len(self.source_data)
        else:
            return self.limit


class TokenizedTranslationDatasetOnTheFly:
    """
    :returns
      - tokenized_source: a list of token in an element of train/val source dataset
      - tokenized_target: a list of token in an element of train/val target dataset
    """
    def __init__(self, ctx, phase, limit=None):
        self.context = ctx
        self.logger = ctx.logger
        self.raw_dataset = TranslationDatasetOnTheFly(ctx, phase, limit)

    def __getitem__(self, item):
        raw_source, raw_target = self.raw_dataset[item]
        tokenized_source = raw_source.split()
        tokenized_target = raw_target.split()
        return tokenized_source, tokenized_target

    def __len__(self):
        return len(self.raw_dataset)


class TokenizedTranslationDataset:
    """
    :returns
        - tokenized_source: a list of token in an item of train/val source dataset
        - tokenized_target: a list of token in an item of train/val target dataset
    """
    def __init__(self, ctx, phase, limit=None):
        self.logger = ctx.logger
        self.raw_dataset = TranslationDataset(ctx, phase, limit)

    def __getitem__(self, item):
        raw_source, raw_target = self.raw_dataset[item]
        tokenized_source = raw_source.split()
        tokenized_target = raw_target.split()
        return tokenized_source, tokenized_target

    def __len__(self):
        return len(self.raw_dataset)


class InputTargetTranslationDatasetOnTheFly:
    """
    :returns
       - tokenized_source: a list of tokens in an item of train/val source dataset
       - tokenized_target: a list of tokens in an item of train/val target dataset
       - inputs: a list of tokens in an item (shift right one position) of train/val source dataset,
                 insert a START_TOKEN in front of the head of [[tokenized_target]]
       - targets: a list of tokens in an item of train/val source dataset,
                  append a END_TOKEN in the tail of [[tokenized_target]]
    """
    def __init__(self, ctx, phase, limit=None):
        self.context = ctx
        self.tokenized_dataset = TokenizedTranslationDatasetOnTheFly(ctx, phase, limit)

    def __getitem__(self, item):
        tokenized_source, tokenized_target = self.tokenized_dataset[item]
        full_target = [START_TOKEN] + tokenized_target + [END_TOKEN]
        """
        a = [START_TOKEN, 2, 3, 4, 5, END_TOKEN]
        a[:-1] = [END_TOKEN, 2, 3, 4, 5]
        a[1:] = [2, 3, 4, 5, 6END_TOKEN]
        """
        inputs = full_target[:-1]
        targets = full_target[1:]
        return tokenized_source, inputs, targets

    def __len__(self):
        return len(self.tokenized_dataset)


class InputTargetTranslationDataset:
    """
        :returns
           - tokenized_source: a list of tokens in an item of train/val source dataset
           - tokenized_target: a list of tokens in an item of train/val target dataset
           - inputs: a list of tokens in an item (shift right one position) of train/val source dataset,
                     insert a START_TOKEN in front of the head of [[tokenized_target]]
           - targets: a list of tokens in an item of train/val source dataset,
                      append a END_TOKEN in the tail of [[tokenized_target]]
    """
    def __init__(self, ctx, phase, limit=None):
        self.context = ctx
        self.tokenized_dataset = TokenizedTranslationDataset(ctx, phase, limit)

    def __getitem__(self, item):
        tokenized_source, tokenized_target = self.tokenized_dataset[item]
        full_target = [START_TOKEN] + tokenized_target + [END_TOKEN]
        """
        a = [START_TOKEN, 2, 3, 4, 5, END_TOKEN]
        a[:-1] = [END_TOKEN, 2, 3, 4, 5]
        a[1:] = [2, 3, 4, 5, 6END_TOKEN]
        """
        inputs = full_target[:-1]
        targets = full_target[1:]
        return tokenized_source, inputs, targets

    def __len__(self):
        return len(self.tokenized_dataset)


class IndexedInputTargetTranslationDatasetOnTheFly:
    """
    :returns
     - indexed_source, a list of index number of tokens in an item in train/val source dataset
     - indexed_inputs, a list of index number of tokens in an item in train/val input dataset
     - indexed_targets, a list of index number of tokens in an item in train/val target dataset
    """
    def __init__(self, ctx, phase, src_dictionary, tgt_dictionary, limit=None):
        self.context = ctx
        self.input_target_dataset = InputTargetTranslationDatasetOnTheFly(ctx, phase, limit)
        self.source_dictionary = src_dictionary
        self.target_dictionary = tgt_dictionary

    def __getitem__(self, item):
        source, inputs, targets = self.input_target_dataset[item]
        indexed_source = self.source_dictionary.index_sentence(source)
        indexed_inputs = self.target_dictionary.index_sentence(inputs)
        indexed_targets = self.target_dictionary.index_sentence(targets)

        return indexed_source, indexed_inputs, indexed_targets

    def __len__(self):
        return len(self.input_target_dataset)

    @staticmethod
    def preprocess(src_dictionary):

        def preprocess_function(source):
            source_tokens = source.strip().split()
            indexed_source = src_dictionary.index_sentence(source_tokens)
            return indexed_source

        return preprocess_function


class IndexedInputTargetTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, ctx, phase):
        self.context = ctx
        # [(indexed_sources, indexed_inputs, indexed_targets), (indexed_sources, indexed_inputs, indexed_targets)]
        self.data = []
        self.phase = phase
        self.index_path = join(ctx.project_processed_dir, f'indexed-{phase}.txt')
        self.vocabulary_size = ctx.vocabulary_size
        self.limit = ctx.dataset_limit

        if not exists(self.index_path):
            # source vocabulary dictionary
            src_dictionary = IndexDictionary.load(ctx.project_processed_dir, mode='source')

            # target vocabulary dictionary
            tgt_dictionary = IndexDictionary.load(ctx.project_processed_dir, mode='target')

            def join_indexes(indexes):
                return ' '.join(str(index) for index in indexes)

            input_target_dataset = InputTargetTranslationDataset(self.context, phase)
            with open(self.index_path, 'w') as file:
                for sources, inputs, targets in input_target_dataset:
                    indexed_sources = join_indexes(src_dictionary.index_sentence(sources))
                    indexed_inputs = join_indexes(tgt_dictionary.index_sentence(inputs))
                    indexed_targets = join_indexes(tgt_dictionary.index_sentence(targets))
                    file.write(f'{indexed_sources}\t{indexed_inputs}\t{indexed_targets}\n')

        def unknownify(index):
            return index if index < self.vocabulary_size else UNK_INDEX
        # unknownify = lambda index: index if index < vocabulary_size else UNK_INDEX
        with open(self.index_path) as file:
            for line in file:
                sources, inputs, targets = line.strip().split('\t')
                if self.vocabulary_size is not None:
                    indexed_sources = [unknownify(int(index)) for index in sources.strip().split(' ')]
                    indexed_inputs = [unknownify(int(index)) for index in inputs.strip().split(' ')]
                    indexed_targets = [unknownify(int(index)) for index in targets.strip().split(' ')]
                else:
                    indexed_sources = [int(index) for index in sources.strip().split(' ')]
                    indexed_inputs = [int(index) for index in inputs.strip().split(' ')]
                    indexed_targets = [int(index) for index in targets.strip().split(' ')]
                self.data.append((indexed_sources, indexed_inputs, indexed_targets))
                if self.limit is not None and len(self.data) >= self.limit:
                    break

    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()

        indexed_sources, indexed_inputs, indexed_targets = self.data[item]
        return indexed_sources, indexed_inputs, indexed_targets

    def __len__(self):
        if self.limit is None:
            return len(self.data)
        else:
            return self.limit

    @staticmethod
    def preprocess(src_dictionary):

        def preprocess_function(source):
            source_tokens = source.strip().split()
            indexed_source = src_dictionary.index_sentence(source_tokens)
            return indexed_source

        return preprocess_function

    @staticmethod
    def prepare(ctx, src_dictionary=None, tgt_dictionary=None):
        """
        :param ctx:
        :param src_dictionary: source vocabulary dictionary
        :param tgt_dictionary: target vocabulary dictionary
        :return:
        """

        if src_dictionary is None:
            # source vocabulary dictionary
            src_dictionary = IndexDictionary.load(ctx.project_processed_dir, mode='source')

        if tgt_dictionary is None:
            # target vocabulary dictionary
            tgt_dictionary = IndexDictionary.load(ctx.project_processed_dir, mode='target')

        def join_indexes(indexes):
            return ' '.join(str(index) for index in indexes)

        for phase in ('train', 'val'):
            input_target_dataset = InputTargetTranslationDataset(ctx, phase)

            with open(join(ctx.project_processed_dir, f'indexed-{phase}.txt'), 'w') as file:
                for sources, inputs, targets in input_target_dataset:
                    indexed_sources = join_indexes(src_dictionary.index_sentence(sources))
                    indexed_inputs = join_indexes(tgt_dictionary.index_sentence(inputs))
                    indexed_targets = join_indexes(tgt_dictionary.index_sentence(targets))
                    file.write(f'{indexed_sources}\t{indexed_inputs}\t{indexed_targets}\n')


if __name__ == "__main__":
    context = Context(desc="dataset")
    logger = context.logger

    if logger.isEnabledFor(logging.DEBUG):
        # Preparing Raw train/val dataset: a file of each line (src, tgt)
        # src-train.txt + tgt-train.txt --> raw-train.txt
        # src-val.txt + tgt-val.txt --> raw-val.txt
        logger.debug("The raw train and validate datasets are generating ...")
        TranslationDataset.prepare(context)

    # a list of train dataset: [(src, tgt), ..., (src, tgt)], build from raw-train.txt
    logger.info("The train dataset [(src, tgt), ..., (src, tgt)] is generating ...")
    translation_dataset = TranslationDataset(context, 'train')

    if logger.isEnabledFor(logging.DEBUG):
        # a list of train dataset: [(src, tgt), ..., (src, tgt)], build from src-train.txt, tgt-train.txt
        logger.debug("The train dataset [(src, tgt), ..., (src, tgt)] is generating on the fly ...")
        translation_dataset_on_the_fly = TranslationDatasetOnTheFly(context, 'train')

        # These datasets should be equal in content
        assert translation_dataset[0] == translation_dataset_on_the_fly[0]

    # a list of train token datasets: [([src_token], [tgt_token]), ..., ([src_token], [tgt_token])]
    # Build it from raw-train.txt
    logger.info("The tokenize train dataset [([token], [token]), ..., ([token], [token])] is generating ...")
    tokenized_dataset = TokenizedTranslationDataset(context, 'train')

    logger.info("The source and target vocabulary dictionaries are generating and saving ...")
    if context.share_dictionary:
        source_generator = shared_tokens_generator(tokenized_dataset)
        source_dictionary = IndexDictionary(source_generator, mode='source')
        # Save source vocabulary
        source_dictionary.save(context.project_processed_dir)

        target_generator = shared_tokens_generator(tokenized_dataset)
        target_dictionary = IndexDictionary(target_generator, mode='target')
        # Save target vocabulary
        target_dictionary.save(context.project_processed_dir)
    else:
        source_generator = source_tokens_generator(tokenized_dataset)
        source_dictionary = IndexDictionary(source_generator, mode='source')
        # Save source vocabulary
        source_dictionary.save(context.project_processed_dir)

        target_generator = target_tokens_generator(tokenized_dataset)
        target_dictionary = IndexDictionary(target_generator, mode='target')
        # Save target vocabulary
        target_dictionary.save(context.project_processed_dir)

    if logger.isEnabledFor(logging.DEBUG):
        # source vocabulary dictionary
        logger.debug("Loading Source Dictionary from vocabulary-source.txt")
        source_dictionary = IndexDictionary.load(context.project_processed_dir, mode='source')

        # target vocabulary dictionary
        logger.debug("Loading Target Dictionary from vocabulary-target.txt")
        target_dictionary = IndexDictionary.load(context.project_processed_dir, mode='target')

    if logger.isEnabledFor(logging.DEBUG):
        logger.info("Convert tokens into index for train/validate datasets ...")
        IndexedInputTargetTranslationDataset.prepare(context, source_dictionary, target_dictionary)

    indexed_translation_dataset = IndexedInputTargetTranslationDataset(context, 'train')
    if logger.isEnabledFor(logging.DEBUG):
        indexed_translation_dataset_on_the_fly = IndexedInputTargetTranslationDatasetOnTheFly(context,
                                                                                              'train',
                                                                                              source_dictionary,
                                                                                              target_dictionary)
        assert indexed_translation_dataset[0] == indexed_translation_dataset_on_the_fly[0]

    logger.info('Done datasets preparation.')
