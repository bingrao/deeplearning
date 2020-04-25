from argument import get_config
from utils.log import get_logger


class Context:
	def __init__(self, desc="Transformer", config=None, logger=None):
		self.description = desc

		# A dictionary of Config Parameters
		if config is None:
			self.config = get_config(desc=self.description)
		else:
			self.config = config

		# logger interface
		if logger is None:
			self.logger = get_logger(self.description, self.getConfig("save_log"))
		else:
			self.logger = logger

		self.logger.info("The Input Parameters:")
		for key, val in self.config.items():
			self.logger.info(f"{key} => {val}")


	def getConfig(self, key):
		return self.config[key]

	def info(self, *args):
		self.logger.info(args)

	def warning(self, *args):
		self.logger.warning(args)

	def error(self, *args):
		self.logger.error(args)

	def debug(self, *args):
		self.logger.debug(args)

	def isEnabledFor(self, level):
		return self.logger.isEnabledFor(level)