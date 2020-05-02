import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
from nmt.data.batch import Batch, run_epoch
from nmt.model.transformer.model import build_model
from nmt.utils.context import Context


class NoamOpt:
	"""Optim wrapper that implements rate."""

	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0

	def step(self):
		"""Update parameters and rate"""
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()

	def rate(self, step=None):
		"""Implement `lrate` above"""
		if step is None:
			step = self._step
		return self.factor * \
			   (self.model_size ** (-0.5) *
				min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
	return NoamOpt(model.context.d_model, 2, 4000,
				   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleLossCompute:
	"""A simple loss compute and train function."""

	def __init__(self, generator, criterion, devices=None, opt=None):
		self.generator = generator
		self.criterion = criterion
		self.opt = opt

	def __call__(self, x, y, norm):
		x = self.generator(x)
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
							  y.contiguous().view(-1)) / norm
		loss.backward()
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()
		# https://github.com/pytorch/pytorch/issues/15585
		# return loss.data[0] * norm
		return loss.data.item() * norm


class LabelSmoothing(nn.Module):
	"""Implement label smoothing."""

	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None

	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, Variable(true_dist, requires_grad=False))


def data_gen(voc_size, batch, nbatches, seq_len = 15):
	"""Generate random data for a src-tgt copy task."""
	for i in range(nbatches):
		# (batch_size, seq_len)
		data = torch.from_numpy(
			np.random.randint(1, voc_size, size=(batch, seq_len)))
		data[:, 0] = 1 # add start token
		src = Variable(data, requires_grad=False)
		tgt = Variable(data, requires_grad=False)
		yield Batch(src, tgt, 0)  # Accessed by next function one by one


if __name__ == "__main__":
	# Train the simple copy task.
	ctx = Context(desc="Train")
	logger = ctx.logger
	vocab_size = 11  # V_Size
	criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
	model = build_model(ctx, vocab_size, vocab_size)
	logger.info(model)
	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
						torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	for epoch in range(100):
		logger.debug("Training Epoch %d", epoch)
		model.train()
		run_epoch(data_gen(vocab_size, 30, 20),
				  model,
				  SimpleLossCompute(model.generator, criterion, model_opt),
				  ctx)

		model.eval()
		run_epoch(data_gen(vocab_size, 30, 5),
				  model,
				  SimpleLossCompute(model.generator, criterion, None),
				  ctx)
