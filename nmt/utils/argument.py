from argparse import ArgumentParser
import torch


def get_main_argument(desc='Train Transformer'):
    parser = ArgumentParser(description=desc)
    # Command Project Related Parameters
    parser.add_argument('--project_name', type=str, required=True, default="")
    parser.add_argument('--project_raw_dir', type=str, required=True, default="")
    parser.add_argument('--project_processed_dir', type=str, required=True, default="")
    parser.add_argument('--project_config', type=str, required=True, default="")
    parser.add_argument('--project_log', type=str, required=True, default="")
    parser.add_argument('--project_save_config', type=bool, default=False)
    parser.add_argument('--project_checkpoint', type=str, default="")
    parser.add_argument('--phase', type=str, required=True, choices=['preprocess', 'train', 'predict', 'val'], default='train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device_id', type=list, default=[0])
    # Predict input source sentence
    parser.add_argument('--source', type=str, default="I am a chinese.")
    parser.add_argument('--save_result', type=str, default=None)
    parser.add_argument('--debug', type=str, default="False")

    # Prediction & Evalation
    # parser.add_argument('--share_dictionary', type=bool, default=False)
    # parser.add_argument('--num_candidates', type=int, default=3)

    # Model & Train Parameters
    # parser.add_argument('--print_every', type=int, default=1000)
    # parser.add_argument('--save_every', type=int, default=10000)
    # parser.add_argument('--dataset_limit', type=int, default=100)
    # parser.add_argument('--vocabulary_size', type=int, default=1000)
    # parser.add_argument('--positional_encoding', action='store_true')
    # parser.add_argument('--d_model', type=int, default=128)
    # parser.add_argument('--layers_count', type=int, default=1)
    # parser.add_argument('--heads_count', type=int, default=2)
    # parser.add_argument('--d_ff', type=int, default=1)
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--label_smoothing', type=float, default=0.1)
    # parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--clip_grads', action='store_true')
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()

    config = vars(args)
    return config
