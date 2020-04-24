import json
from argparse import ArgumentParser


def get_config(desc="transformer"):
	parser = ArgumentParser(description=desc)
	parser.add_argument('--config',
						type=str,
						required=True,
						default='configs/config_translate.json')

	# #%%%%%%%%%%%%%%%%%%%% Training Task
	# parser.add_argument('--data_dir', type=str, default='data/example/processed')
	# parser.add_argument('--save_config', type=str, default=None)
	# parser.add_argument('--save_checkpoint', type=str, default=None)
	# parser.add_argument('--save_log', type=str, default=None)
	#
	# parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	#
	# parser.add_argument('--dataset_limit', type=int, default=None)
	# parser.add_argument('--print_every', type=int, default=1)
	# parser.add_argument('--save_every', type=int, default=1)
	#
	# parser.add_argument('--vocabulary_size', type=int, default=None)
	# parser.add_argument('--positional_encoding', action='store_true')
	#
	# parser.add_argument('--d_model', type=int, default=32)
	# parser.add_argument('--layers_count', type=int, default=6)
	# parser.add_argument('--heads_count', type=int, default=8)
	# parser.add_argument('--d_ff', type=int, default=64)
	# parser.add_argument('--dropout_prob', type=float, default=0.1)
	#
	# parser.add_argument('--label_smoothing', type=float, default=0.1)
	# parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
	# parser.add_argument('--lr', type=float, default=0.001)
	# parser.add_argument('--clip_grads', action='store_true')

	# parser.add_argument('--batch_size', type=int, default=64)
	# parser.add_argument('--epochs', type=int, default=100)

	# #%%%%%%%%%%%%%%%%%%%% Preparing Dataset Task
	# parser.add_argument('--train_source', type=str, default='data/example/raw/src-train.txt')
	# parser.add_argument('--train_target', type=str, default='data/example/raw/tgt-train.txt')
	# parser.add_argument('--val_source', type=str, default='data/example/raw/src-val.txt')
	# parser.add_argument('--val_target', type=str, default='data/example/raw/tgt-val.txt')
	# parser.add_argument('--save_data_dir', type=str, default='data/example/processed')
	# parser.add_argument('--share_dictionary', type=bool, default=False)

	# #%%%%%%%%%%%%%%%%%%%% Evaluate Task
	# parser.add_argument('--save_result', type=str, default='logs/example_eval.txt')
	# parser.add_argument('--config', type=str, required=True, default='checkpoints/example_config.json')
	# parser.add_argument('--checkpoint', type=str, required=True, default='checkpoints/example_model.pth')
	# parser.add_argument('--phase', type=str, default='val', choices=['train', 'val'])

	args = parser.parse_args()

	if args.config is not None:
		with open(args.config) as f:
			config = json.load(f)

		default_config = vars(args)
		for key, default_value in default_config.items():
			if key not in config:
				config[key] = default_value
	else:
		config = vars(args)  # convert to dictionary

	return config
