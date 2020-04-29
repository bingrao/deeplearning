from nmt.utils.argument import get_config
from nmt.utils.log import get_logger
from os.path import join, exists
from os import makedirs
import os
import torch


def create_dir(dir_path):
	if not exists(dir_path):
		makedirs(dir_path)


class Context:
	def __init__(self, desc="Transformer", config=None, logger=None):
		self.description = desc

		# A dictionary of Config Parameters
		self.config = get_config(desc=self.description) if config is None else config

		self.proj_name = self.config["project_name"]

		self.project_raw_dir = str(self.config["project_raw_dir"])
		create_dir(self.project_raw_dir)

		self.train_src_dataset = self.project_raw_dir + "/src-train.txt"
		self.train_dst_dataset = self.project_raw_dir + "/tgt-train.txt"
		self.val_src_dataset = self.project_raw_dir + "/src-val.txt"
		self.val_dst_dataset = self.project_raw_dir + "/tgt-val.txt"
		self.test_src_dataset = self.project_raw_dir + "/src-test.txt"

		self.project_processed_dir = self.config["project_processed_dir"]
		create_dir(self.project_processed_dir)

		self.project_config = self.config["project_config"]
		if not exists(self.project_config):
			create_dir(os.path.dirname(self.project_config))

		self.project_log = self.config["project_log"]
		if not exists(self.project_log):
			create_dir(os.path.dirname(self.project_log))

		self.project_checkpoint = str(self.config["project_checkpoint"])
		if not exists(self.project_checkpoint):
			create_dir(os.path.dirname(self.project_checkpoint))

		# logger interface
		self.logger = get_logger(self.description, self.project_log) if logger is None else logger

		# Model Paramters
		self.phase = self.config["phase"]
		self.d_model = self.config["d_model"]
		self.layers_count = self.config["layers_count"]
		self.heads_count = self.config["heads_count"]
		self.d_ff = self.config["d_ff"]
		self.dropout = self.config["dropout"]
		self.label_smoothing = self.config["label_smoothing"]
		self.optimizer = self.config["optimizer"]
		self.lr = self.config["lr"]
		self.clip_grads = self.config["clip_grads"]
		self.batch_size = self.config["batch_size"]
		self.epochs = self.config["epochs"]
		self.vocabulary_size = self.config["vocabulary_size"]

		self.dataset_limit = self.config["dataset_limit"]
		self.print_every = self.config["print_every"]
		self.print_every = self.config["print_every"]

		self.pred_source = self.config["source"]
		self.num_candidates = self.config["num_candidates"]
		self.save_result = self.config["save_result"]
		self.share_dictionary = self.config["share_dictionary"]
		self.save_every = self.config["save_every"]



		# Trainning Device Set up
		self.device = torch.device(self.config["device"])
		self.device_id = list(self.config["device_id"])
		self.is_cuda = self.config["device"] == 'cuda'
		self.is_cpu = self.config["device"] == 'cpu'
		self.is_gpu_parallel = self.is_cuda and (len(self.device_id) > 1)

		self.logger.info("The Input Parameters:")
		for key, val in self.config.items():
			self.logger.info(f"{key} => {val}")
