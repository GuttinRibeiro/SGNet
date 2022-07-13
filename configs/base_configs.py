import argparse

__all__ = ['parse_base_args']

def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--anneal', default='sigmoid', type=str)
    parser.add_argument('--increase_beta_after', default=1, type=int)
    parser.add_argument('--anneal_start', default=0.0, type=float)
    parser.add_argument('--anneal_finish', default=1.0, type=float)
    parser.add_argument('--anneal_center_step', default=10.0, type=float)
    parser.add_argument('--anneal_steps', default=1000.0, type=float)
    parser.add_argument('--anneal_period', default='step', type=str)
    parser.add_argument('--early_stopping', default=False, action='store_true')
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--save_results', default=False, action='store_true')
    parser.add_argument('--save_directory', default='results', type=str)
    return parser

