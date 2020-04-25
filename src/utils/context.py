from utils.argument import get_config
from utils.log import get_logger
from os.path import join, exists
from os import makedirs


class Context:
	def __init__(self, desc="Transformer", config=None, logger=None):
		self.description = desc

		# A dictionary of Config Parameters
		if config is None:
			self.config = get_config(desc=self.description)
		else:
			self.config = config

		self.save_data_dir = self.config["save_data_dir"]
		if not exists(self.save_data_dir):
			makedirs(self.save_data_dir)

		self.checkpoint_dir = join(self.save_data_dir, 'checkpoints')
		if not exists(self.checkpoint_dir):
			makedirs(self.checkpoint_dir)

		self.log_dir = join(self.save_data_dir, 'logs')
		if not exists(self.log_dir):
			makedirs(self.log_dir)

		# logger interface
		if logger is None:
			self.logger = get_logger(self.description, self.config["save_log"])
		else:
			self.logger = logger

		self.logger.info("The Input Parameters:")
		for key, val in self.config.items():
			self.logger.info(f"{key} => {val}")