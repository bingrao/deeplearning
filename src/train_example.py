# from models import build_model
# from torchtext import data, datasets
# from losses import SimpleLossCompute
# from optimizers import NoamOpt
# from utils.log import get_logger
# from argument import get_config
# import numpy as np
# import torch
# import torch.nn as nn
# import time
# from torch.autograd import Variable
# from torchtext import data
#
# logger = get_logger("train_example")
# global max_src_in_batch, max_tgt_in_batch
#
#
# def data_gen(V, batch, nbatches):
#     """Generate random data for a src-tgt copy task."""
#     for i in range(nbatches):
#         dataset = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
#         dataset[:, 0] = 1
#         src = Variable(dataset, requires_grad=False)
#         tgt = Variable(dataset, requires_grad=False)
#         yield Batch(src, tgt, 0)
#
#
# def train_simple_with_dummy_data(V=11):
#     # Train the simple copy task.
#     criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
#     logger.info(criterion)
#     model = build_model(get_config(logger=get_logger()), V, V)
#     logger.info(model)
#     model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
#                         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#
#     for epoch in range(10):
#         model.train()
#         run_epoch(data_gen(V, 30, 20), model,
#                   SimpleLossCompute(model.generator, criterion, model_opt))
#         model.eval()
#         logger.info(run_epoch(data_gen(V, 30, 5), model,
#                         SimpleLossCompute(model.generator, criterion, None)))
#
#     model.eval()
#     src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
#     src_mask = Variable(torch.ones(1, 1, 10))
#     logger.info(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
#
#
# if __name__ == "__main__":
#     # train_with_spacy_dataset()
#     train_simple_with_dummy_data()