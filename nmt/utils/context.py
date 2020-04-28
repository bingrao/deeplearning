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
		if config is None:
			self.config = get_config(desc=self.description)
		else:
			self.config = config

		self.proj_name = self.config["project_name"]

		self.proj_raw_dir = str(self.config["project_raw_dir"])
		create_dir(self.proj_raw_dir)

		self.train_src_dataset = self.proj_raw_dir + "/src-train.txt"
		self.train_dst_dataset = self.proj_raw_dir + "/tgt-train.txt"
		self.val_src_dataset = self.proj_raw_dir + "/src-val.txt"
		self.val_dst_dataset = self.proj_raw_dir + "/tgt-val.txt"
		self.test_src_dataset = self.proj_raw_dir + "/src-test.txt"

		self.proj_processed_dir = str(self.config["project_processed_dir"])
		create_dir(self.proj_processed_dir)

		self.project_config = str(self.config["project_config"])
		if not exists(self.project_config):
			create_dir(os.path.dirname(self.project_config))

		self.project_log = str(self.config["project_log"])
		if not exists(self.project_log):
			create_dir(os.path.dirname(self.project_log))

		self.project_checkpoint = str(self.config["project_checkpoint"])
		if not exists(self.project_checkpoint):
			create_dir(os.path.dirname(self.project_checkpoint))

		# logger interface
		if logger is None:
			self.logger = get_logger(self.description, self.project_log)
		else:
			self.logger = logger

		self.logger.info("The Input Parameters:")
		for key, val in self.config.items():
			self.logger.info(f"{key} => {val}")

		self.device = torch.device(self.config["device"])
		self.device_id = list(self.config["device_id"])
		self.is_cuda = self.config["device"] == 'cuda'
		self.is_cpu = self.config["device"] == 'cpu'
		self.is_gpu_parallel = self.is_cuda and (len(self.device_id) > 1)
