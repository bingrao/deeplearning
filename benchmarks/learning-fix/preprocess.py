from torch.utils.data import Dataset
from nmt.utils.context import Context
from os.path import join
from nmt.data.batch import Batch
from torch.autograd import Variable
import collections
import torch
START_TOKEN = '<Start>'
END_TOKEN = '<End>'
UNK_TOKEN = 1000

# class Batch:
# 	def __init__(self, src, tgt=None, pad=0):
# 		self.src = src
# 		self.src_mask = (self.src != pad).unsqueeze(-2)
# 		if tgt is not None:
# 			self.tgt = tgt
# 			self.tgt_in = self.tgt[:, :-1]  # remove last column
# 			self.tgt_out = self.tgt[:, 1:]  # remove first column
# 			self.tgt_mask = make_std_mask(self.tgt_in, pad)
# 			self.ntokens = (self.tgt_out != pad).data.sum()


class dataset(Dataset):
	def __init__(self,
				 ctx=None,
				 target="train",
				 dataset="small",
				 embedding=None,
				 padding= 0):
		assert target != "train" or target != "eval" or target != "test"
		self.context = ctx
		self.logger = ctx.logger
		self.target = target
		self.raw_dir = join(self.context.project_raw_dir, dataset)
		self.processed_dir = self.context.project_processed_dir
		self.padding = padding
		self.min_frequency = 0
		self.max_vocab_size = 250
		self.src_vocab = self._build_vocab(obj="buggy",
										   min_frequency=self.min_frequency,
										   max_vocab_size=self.max_vocab_size)
		self.tgt_vocab = self._build_vocab(obj="fixed",
										   min_frequency=self.min_frequency,
										   max_vocab_size=self.max_vocab_size)

		self.token_embedding = embedding

		self.data = []
		# load data
		self.load()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		# Return a batch representation with
		# src, tgt_in, tgt_out and masks
		return self.data[idx]

	def load(self):
		src_path = join(self.raw_dir, self.target, "buggy.txt")
		tgt_path = join(self.raw_dir, self.target, "fixed.txt")
		raw_data_path = join(self.processed_dir, f'{self.target}-raw.txt')
		raw_token_path = join(self.processed_dir, f'{self.target}-token.txt')
		raw_idx_path = join(self.processed_dir, f'{self.target}-index.txt')

		with open(src_path) as src_file:
			src_raw_data = src_file.readlines()
		with open(tgt_path) as tgt_file:
			tgt_raw_data = tgt_file.readlines()


		with open(raw_data_path, 'w') as f1, \
				open(raw_token_path, 'w') as f2, \
				open(raw_idx_path, 'w') as f3:
			for src_line, tgt_line in zip(src_raw_data, tgt_raw_data):
				src_tokens = self.to_tokenize(src_line)
				src_index = self.to_embedding(src_tokens, obj="buggy")
				src_feature = Variable(torch.tensor(src_index).unsqueeze(0),
							   requires_grad=False)

				tgt_tokens = [START_TOKEN] + self.to_tokenize(tgt_line) + [END_TOKEN]
				tgt_index = self.to_embedding(tgt_tokens, obj="fixed")
				tgt_feature = Variable(torch.tensor(tgt_index).unsqueeze(0),
							   requires_grad=False)
				if self.context.isDebug:
					f1.write(f'{src_line}\t{tgt_line}\n')
					f2.write(f'{src_tokens}\t{tgt_tokens}\n')
					f3.write(f'{src_index}\t{tgt_index}\n')

				self.data.append(Batch(src_feature, tgt_feature, self.padding))

	def _build_vocab(self,
					 obj="buggy",
					 min_frequency=0,
					 max_vocab_size=250,
					 downcase=False,
					 delimiter=" "):

		assert obj != "buggy" or obj != "fixed"

		# Counter for all tokens in the vocabulary
		cnt = collections.Counter()
		source_path = join(self.raw_dir, "train", f"{obj}.txt")
		with open(source_path) as file:
			for line in file:
				if downcase:
					line = line.lower()
				if delimiter == "":
					tokens = list(line.strip())
				else:
					tokens = line.strip().split(delimiter)
				tokens = [_ for _ in tokens if len(_) > 0]
				cnt.update(tokens)

		self.logger.info("Found %d unique tokens in the vocabulary.", len(cnt))

		# Filter tokens below the frequency threshold
		if min_frequency > 0:
			filtered_tokens = [(w, c) for w, c in cnt.most_common() if c > min_frequency]
			cnt = collections.Counter(dict(filtered_tokens))

		self.logger.info("Found %d unique tokens with frequency > %d.", len(cnt), min_frequency)

		# Sort tokens by 1. frequency 2. lexically to break ties
		word_with_counts = cnt.most_common()
		word_with_counts = sorted(word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

		# Take only max-vocab
		if max_vocab_size is not None:
			word_with_counts = word_with_counts[:max_vocab_size]

		if self.context.isDebug:
			with open(join(self.processed_dir, f"vocab.{obj}.txt"), 'w') as file:
				for idx, (word, count) in enumerate(word_with_counts):
					file.write(f'{idx}\t{word}\t{count}\n')

		# ['for', 'int', ... , 'float']
		return [word for word,count in word_with_counts]

	def to_tokenize(self, text):
		return [token for token in text.split()]

	def to_embedding(self, tokens, obj="buggy"):
		assert obj != "buggy" or obj != "fixed"
		if self.token_embedding is not None:
			return [self.token_embedding(token) for token in tokens]
		elif obj == "buggy":
			return [self.src_vocab.index(token)
					if token in self.src_vocab else UNK_TOKEN for token in tokens]
		else:
			return [self.tgt_vocab.index(token)
					if token in self.tgt_vocab else UNK_TOKEN for token in tokens]




if __name__ == "__main__":
	context = Context("Learning-fix based on Transformer")
	train_dataset = dataset(ctx=context, target="train")