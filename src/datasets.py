from os.path import dirname, abspath, join, exists
from os import makedirs
from dictionaries import IndexDictionary
from utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator
from argparse import ArgumentParser
from dictionaries import START_TOKEN, END_TOKEN
from utils.log import get_logger
import logging

UNK_INDEX = 1


class TranslationDataset:
    """
    :return
      - self.data: a list of train/val raw data pairs (src, tgt)
    """
    def __init__(self, config, phase, limit=None):
        assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'"

        self.limit = limit
        self.config = config
        self.phase = phase
        self.raw_data_path = join(config.save_data_dir, f'raw-{phase}.txt')
        self.data = []

        if not exists(config.save_data_dir):
            makedirs(config.save_data_dir)

        if not exists(self.raw_data_path):
            if phase == 'train':
                source_filepath = config.train_source
                target_filepath = config.train_target
            elif phase == 'val':
                source_filepath = config.val_source
                target_filepath = config.val_target
            else:
                raise NotImplementedError()

            with open(source_filepath) as source_file:
                source_data = source_file.readlines()

            with open(target_filepath) as target_filepath:
                target_data = target_filepath.readlines()

            with open(self.raw_data_path, 'w') as file:
                for source_line, target_line in zip(source_data, target_data):
                    source_line = source_line.strip()
                    target_line = target_line.strip()
                    line = f'{source_line}\t{target_line}\n'
                    file.write(line)

        with open(join(config.save_data_dir, f'raw-{phase}.txt')) as file:
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
    def prepare(config):
        """
        1.  Combine train source and target into a single file
        2.  Combine validate source and target into a single file
        """
        if not exists(config.save_data_dir):
            makedirs(config.save_data_dir)

        for phase in ('train', 'val'):

            if phase == 'train':
                source_filepath = config.train_source
                target_filepath = config.train_target
            else:
                source_filepath = config.val_source
                target_filepath = config.val_target

            with open(source_filepath) as source_file:
                source_data = source_file.readlines()

            with open(target_filepath) as target_filepath:
                target_data = target_filepath.readlines()

            with open(join(config.save_data_dir, f'raw-{phase}.txt'), 'w') as file:
                for source_line, target_line in zip(source_data, target_data):
                    source_line = source_line.strip()
                    target_line = target_line.strip()
                    line = f'{source_line}\t{target_line}\n'
                    file.write(line)


class TranslationDatasetOnTheFly:
    """
    :returns
      - self.source_data: a list of train/val raw source data
      - self.target_data: a list of train/val raw target data
    """
    def __init__(self, config, phase, limit=None):
        assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'"

        self.limit = limit

        if phase == 'train':
            source_filepath = config.train_source
            target_filepath = config.train_target
        elif phase == 'val':
            source_filepath = config.val_source
            target_filepath = config.val_target
        else:
            raise NotImplementedError()

        with open(source_filepath) as source_file:
            self.source_data = source_file.readlines()

        with open(target_filepath) as target_filepath:
            self.target_data = target_filepath.readlines()

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
    def __init__(self, config, phase, limit=None):

        self.raw_dataset = TranslationDatasetOnTheFly(config, phase, limit)

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
    def __init__(self, config, phase, limit=None):

        self.raw_dataset = TranslationDataset(config, phase, limit)

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
    def __init__(self, config, phase, limit=None):
        self.tokenized_dataset = TokenizedTranslationDatasetOnTheFly(config, phase, limit)

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
    def __init__(self, config, phase, limit=None):
        self.tokenized_dataset = TokenizedTranslationDataset(config, phase, limit)

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
    def __init__(self, config, phase, src_dictionary, tgt_dictionary, limit=None):

        self.input_target_dataset = InputTargetTranslationDatasetOnTheFly(config, phase, limit)
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


class IndexedInputTargetTranslationDataset:
    def __init__(self, config, phase, vocabulary_size=None, limit=None):

        # [(indexed_sources, indexed_inputs, indexed_targets), (indexed_sources, indexed_inputs, indexed_targets)]
        self.data = []

        def unknownify(index):
            return index if index < vocabulary_size else UNK_INDEX
        # unknownify = lambda index: index if index < vocabulary_size else UNK_INDEX
        with open(join(config.save_data_dir, f'indexed-{phase}.txt')) as file:
            for line in file:
                sources, inputs, targets = line.strip().split('\t')
                if vocabulary_size is not None:
                    indexed_sources = [unknownify(int(index)) for index in sources.strip().split(' ')]
                    indexed_inputs = [unknownify(int(index)) for index in inputs.strip().split(' ')]
                    indexed_targets = [unknownify(int(index)) for index in targets.strip().split(' ')]
                else:
                    indexed_sources = [int(index) for index in sources.strip().split(' ')]
                    indexed_inputs = [int(index) for index in inputs.strip().split(' ')]
                    indexed_targets = [int(index) for index in targets.strip().split(' ')]
                self.data.append((indexed_sources, indexed_inputs, indexed_targets))
                if limit is not None and len(self.data) >= limit:
                    break

        self.vocabulary_size = vocabulary_size
        self.limit = limit

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
    def prepare(config, src_dictionary, tgt_dictionary):
        """

        :param config:
        :param src_dictionary: source vocabulary dictionary
        :param tgt_dictionary: target vocabulary dictionary
        :return:
        """
        def join_indexes(indexes):
            return ' '.join(str(index) for index in indexes)
        # join_indexes = lambda indexes: ' '.join(str(index) for index in indexes)
        for phase in ('train', 'val'):
            input_target_dataset = InputTargetTranslationDataset(config, phase)

            with open(join(config.save_data_dir, f'indexed-{phase}.txt'), 'w') as file:
                for sources, inputs, targets in input_target_dataset:
                    indexed_sources = join_indexes(src_dictionary.index_sentence(sources))
                    indexed_inputs = join_indexes(tgt_dictionary.index_sentence(inputs))
                    indexed_targets = join_indexes(tgt_dictionary.index_sentence(targets))
                    file.write(f'{indexed_sources}\t{indexed_inputs}\t{indexed_targets}\n')


if __name__ == "__main__":
    parser = ArgumentParser('Prepare datasets')
    parser.add_argument('--train_source', type=str, default=None)
    parser.add_argument('--train_target', type=str, default=None)
    parser.add_argument('--val_source', type=str, default=None)
    parser.add_argument('--val_target', type=str, default=None)
    parser.add_argument('--save_data_dir', type=str, default=None)
    parser.add_argument('--share_dictionary', type=bool, default=False)
    args = parser.parse_args()
    logger = get_logger("[Prepare_Dataset]-")

    logger.info(args)

    # Preparing Raw train/val dataset: a file of each line (src, tgt)
    # src-train.txt + tgt-train.txt --> raw-train.txt
    # src-val.txt + tgt-val.txt --> raw-val.txt
    # logger.info("The raw train and validate datasets are generating ...")
    # TranslationDataset.prepare(args)

    # a list of train dataset: [(src, tgt), ..., (src, tgt)], build from raw-train.txt
    logger.info("The train dataset [(src, tgt), ..., (src, tgt)] is generating ...")
    translation_dataset = TranslationDataset(args, 'train')

    if logger.isEnabledFor(logging.DEBUG):
        # a list of train dataset: [(src, tgt), ..., (src, tgt)], build from src-train.txt, tgt-train.txt
        logger.debug("The train dataset [(src, tgt), ..., (src, tgt)] is generating on the fly ...")
        translation_dataset_on_the_fly = TranslationDatasetOnTheFly(args, 'train')

        # These datasets should be equal in content
        assert translation_dataset[0] == translation_dataset_on_the_fly[0]

    # a list of train token datasets: [([src_token], [tgt_token]), ..., ([src_token], [tgt_token])]
    # Build it from raw-train.txt
    logger.info("The tokenize train dataset [([token], [token]), ..., ([token], [token])] is generating ...")
    tokenized_dataset = TokenizedTranslationDataset(args, 'train')

    logger.info("The source and target vocabulary dictionaries are generating and saving ...")
    if args.share_dictionary:

        source_generator = shared_tokens_generator(tokenized_dataset)
        source_dictionary = IndexDictionary(source_generator, mode='source')
        # Save source vocabulary
        source_dictionary.save(args.save_data_dir)

        target_generator = shared_tokens_generator(tokenized_dataset)
        target_dictionary = IndexDictionary(target_generator, mode='target')
        # Save target vocabulary
        target_dictionary.save(args.save_data_dir)
    else:
        source_generator = source_tokens_generator(tokenized_dataset)
        source_dictionary = IndexDictionary(source_generator, mode='source')
        # Save source vocabulary
        source_dictionary.save(args.save_data_dir)

        target_generator = target_tokens_generator(tokenized_dataset)
        target_dictionary = IndexDictionary(target_generator, mode='target')
        # Save target vocabulary
        target_dictionary.save(args.save_data_dir)

    if logger.isEnabledFor(logging.DEBUG):
        # source vocabulary dictionary
        logger.debug("Loading Source Dictionary from vocabulary-source.txt")
        source_dictionary = IndexDictionary.load(args.save_data_dir, mode='source')

        # target vocabulary dictionary
        logger.debug("Loading Target Dictionary from vocabulary-target.txt")
        target_dictionary = IndexDictionary.load(args.save_data_dir, mode='target')

    logger.info("Convert tokens into index for train/validate datasets ...")
    IndexedInputTargetTranslationDataset.prepare(args, source_dictionary, target_dictionary)

    if logger.isEnabledFor(logging.DEBUG):
        indexed_translation_dataset = IndexedInputTargetTranslationDataset(args, 'train')
        indexed_translation_dataset_on_the_fly = IndexedInputTargetTranslationDatasetOnTheFly(args,
                                                                                              'train',
                                                                                              source_dictionary,
                                                                                              target_dictionary)
        assert indexed_translation_dataset[0] == indexed_translation_dataset_on_the_fly[0]

    logger.info('Done datasets preparation.')
