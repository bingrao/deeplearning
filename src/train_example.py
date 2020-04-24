import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import build_model
from torchtext import data, datasets
from train import LabelSmoothing, Batch, run_epoch, MultiGPULossCompute
from train import rebatch, MyIterator, greedy_decode
from losses import SimpleLossCompute
from optimizers import NoamOpt
from utils.log import get_logger
from argument import get_config
global max_src_in_batch, max_tgt_in_batch

logger = get_logger("train_example")


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        dataset = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        dataset[:, 0] = 1
        src = Variable(dataset, requires_grad=False)
        tgt = Variable(dataset, requires_grad=False)
        yield Batch(src, tgt, 0)


def train_simple_with_dummy_data(V=11):
    # Train the simple copy task.
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    logger.info(criterion)
    model = build_model(get_config(), V, V)
    logger.info(model)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        logger.info(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    logger.info(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


##################################################################################
# Now we consider a real-world example using the IWSLT German-English Translation task.
# This task is much smaller than the WMT task considered in the paper, but it illustrates
# the whole system. We also show how to use multi-gpu processing to make it really fast.

# !pip install torchtext spacy
# !python -m spacy download en   --> You can now load the model via spacy.load('en')
# !python -m spacy download de   --> You can now load the model via spacy.load('de')

def train_with_spacy_dataset():
    # We will load the dataset using torchtext and spacy for tokenization.
    if True:
        import spacy

        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"
        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD)

        MAX_LEN = 100
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                  len(vars(x)['trg']) <= MAX_LEN)
        MIN_FREQ = 2
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    # GPUs to use
    devices = [0, 1, 2, 3]
    if True:
        pad_idx = TGT.vocab.stoi["<blank>"]
        model = build_model(get_config(), len(SRC.vocab), len(TGT.vocab))
        model.cuda()
        criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
        criterion.cuda()
        BATCH_SIZE = 12000
        train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True)
        valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)
        model_par = nn.DataParallel(model, device_ids=devices)

    # Training the System
    # !wget https://s3.amazonaws.com/opennmt-models/iwslt.pt

    if True:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(10):
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      MultiGPULossCompute(model.generator, criterion, devices=devices))
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_par,
                             MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
            print(loss)
    else:
        model = torch.load("iwslt.pt")

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break


if __name__ == "__main__":
    # train_with_spacy_dataset()
    train_simple_with_dummy_data()