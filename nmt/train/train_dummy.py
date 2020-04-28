import numpy as np
import torch
from torch.autograd import Variable
from nmt.data.batch import Batch, run_epoch
from nmt.model.transformer.model import build_model
from nmt.utils.context import Context
from nmt.train.optimizers import NoamOpt
from nmt.train.common import LabelSmoothing
from nmt.train.losses import SimpleLossCompute



def data_gen(voc_size, batch, nbatches):
	"""Generate random data for a src-tgt copy task."""
	for i in range(nbatches):
		data = torch.from_numpy(np.random.randint(1, voc_size, size=(batch, 10)))
		data[:, 0] = 1
		src = Variable(data, requires_grad=False)
		tgt = Variable(data, requires_grad=False)
		yield Batch(src, tgt, 0)


if __name__ == "__main__":
	# Train the simple copy task.
	ctx = Context(desc="Train")
	logger = ctx.logger
	vocab_size = 11		# V_Size
	criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
	model = build_model(ctx, vocab_size, vocab_size)
	logger.info(model)
	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
						torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	for epoch in range(100):
		logger.info("Training Epoch %d", epoch)
		model.train()
		run_epoch(data_gen(vocab_size, 30, 20),
				  model,
				  SimpleLossCompute(model.generator, criterion, model_opt),
				  ctx)

		logger.info("Evaluating Epoch %d", epoch)
		model.eval()
		logger.info(run_epoch(data_gen(vocab_size, 30, 5),
							  model,
							  SimpleLossCompute(model.generator, criterion, None),
							  ctx))
