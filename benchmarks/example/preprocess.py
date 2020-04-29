import logging
from nmt.utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator
from nmt.utils.context import Context
from benchmarks.example.datasets import TranslationDataset, TokenizedTranslationDataset
from benchmarks.example.datasets import IndexedInputTargetTranslationDataset
from benchmarks.example.datasets import IndexedInputTargetTranslationDatasetOnTheFly, TranslationDatasetOnTheFly
from benchmarks.example.dictionaries import IndexDictionary

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
