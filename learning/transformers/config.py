import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Transformers for Question-Sparql')

    parser.add_argument('--mode', default='train',
                        help='mode: `train` or `test`')
    parser.add_argument('--data', default='learning/transformers/data/lcquad10/',
                        help='path to dataset')
    parser.add_argument('--glove',default='learning/transformers/data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='learning/transformers/checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--save_model', default=True, action='store_true',
                        help='flag to save model')
    parser.add_argument('--load_model', default=False, action='store_false',
                        help='flag to load model')
    parser.add_argument('--checkpoint_path', default='learning/transformers/checkpoints/',
                        help='directory to save/load checkpoints in')
    parser.add_argument('--checkpoint_name', type=str, default='lcquad10',
                        help='Name to identify experiment')

    # parser.add_argument('--my_variable', default=False, action='store_true')
    # parser.add_argument('--other_variable', default=True, action='store_false')

    parser.add_argument('--load', default='learning/transformers/checkpoints/',
                        help='directory to load checkpoints in')
    parser.add_argument('--expname', type=str, default='lcquad10',
                        help='Name to identify experiment')
    # model arguments
    parser.add_argument('--input_dim', default=300, type=int,
                        help='Size of input word vector')
    parser.add_argument('--mem_dim', default=150, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden_dim', default=50, type=int,
                        help='Size of classifier MLP')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of classes in dataset')
    # training arguments
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=0.00225, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--sim', default='nn',
                        help='similarity (default: nn) nn or cos')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
