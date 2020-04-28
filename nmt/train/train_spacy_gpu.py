import torch.nn as nn
from torch.autograd import Variable
import torch
from nmt.model.transformer.model import build_model
from nmt.utils.context import Context
from nmt.train.common import LabelSmoothing
from nmt.data.batch import batch_size_fn, run_epoch, rebatch
from nmt.utils.pad import subsequent_mask
from nmt.train.optimizers import NoamOpt
from nmt.data.preprocess import MyIterator


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# Finally to really target fast training, we will use multi-gpu.
# This code implements multi-gpu word generation. It is not specific to
# transformer so I won’t go into too much detail. The idea is to split up
# word generation at training time into chunks to be processed in parallel
# across many different gpus. We do this using pytorch parallel primitives:

# replicate - split modules onto different gpus.
# scatter - split batches onto different gpus
# parallel_apply - apply module to batches on different gpus
# gather - pull scattered data back onto one gpu.
# nn.DataParallel - a special module wrapper that calls these all before evaluating.

class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=self.devices)
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions  requires_grad=self.opt is not None)]
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=True)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum().item() / normalize
            total += l.data.item()

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


if __name__ == "__main__":
    ctx = Context("Train_MultiGPU")
    config = ctx.config
    logger = ctx.logger
    BATCH_SIZE = config["batch_size"]
    epochs = config["epochs"]

    # For data loading.
    from torchtext import data, datasets
    logger.info("Preparing dataset ...")
    if True:    # Preparing dataset
        import spacy

        # !pip install torchtext spacy
        # !python -m spacy download en
        # !python -m spacy download de
        logger.info("Load en/de data from local ...")
        spacy_de = spacy.load('de', path=ctx.proj_raw_dir)
        spacy_en = spacy.load('en', path=ctx.proj_raw_dir)

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        # tokenize_en("I am a Chinese")  --> ['I', 'am', 'a', 'Chinese']
        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        logger.info("Build SRC and TGT Fields ...")
        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"
        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                         eos_token=EOS_WORD, pad_token=BLANK_WORD)
        logger.info("Split datasets into train, val and test using SRC/TGT fileds ...")
        MAX_LEN = 100
        # Spilt dataset in root path into train, val, and test dataset
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'),  # A tuple containing the extension to path for each language.
            fields=(SRC, TGT),  # A tuple containing the fields that will be used for data in each language.
            root=ctx.proj_raw_dir,  # Root dataset storage directory.
            # train='train',  # The prefix of the train data.
            # validation='val',  # The prefix of the validation data
            # test='test',  # The prefix of the test data. Default
            # Filter condition to extract only datasets satifying pred
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
        MIN_FREQ = 2
        logger.info("Build vocabularies for src and tgt ...")
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    # GPUs to use
    devices = [0]   # [0, 1, 2, 3]
    pad_idx = TGT.vocab.stoi["<blank>"]
    logger.info("Build Model ...")
    model = build_model(ctx, len(SRC.vocab), len(TGT.vocab))

    logger.info(model)

    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Generating Training and Validating Batch datasets ...")
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=cuda_device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=cuda_device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    logger.info("Training Process is begining ...")

    # Training or load model from checkpoint
    if True:
        model_opt = NoamOpt(model.src_embed[0].d_model,
                            1,
                            2000,
                            torch.optim.Adam(model.parameters(),
                                             lr=0,
                                             betas=(0.9, 0.98),
                                             eps=1e-9))
        for epoch in range(epochs):
            # Set model in train
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      MultiGPULossCompute(model.generator, criterion,
                                          devices=devices, opt=model_opt))

            # Evaluation Model
            model_par.eval()

            # Get loss
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                             model_par,
                             MultiGPULossCompute(model.generator, criterion,
                                                 devices=devices, opt=None))
            logger.info("The loss is %d", loss)
    else:
        model = torch.load("iwslt.pt")

    logger.info("Training is over and Evaluate Model  ...")
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>":
                break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break

