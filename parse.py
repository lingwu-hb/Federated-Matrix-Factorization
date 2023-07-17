import argparse
import torch


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--hiddenDim', type=int, default=64, help='hidden state size')
    parser.add_argument('--early_stop', type=int, default=50, help='Patience for early stop')
    parser.add_argument('--epochs', type=int, default=2, help='the number of epochs to train for')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')

    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')
    return args