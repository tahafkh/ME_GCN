import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ME-GCN for Offensive Language Detection')

    parser.add_argument('--word_embedding', type=str, default='word2vec',
                        help='Word embedding type.')
    parser.add_argument('--dim', type=int, default=25,
                        help='Edge embedding dimension')
    parser.add_argument('--threshold', type=int, default=15,
                        help='Threshold for d2d edge weight')
    parser.add_argument('--pooling', type=str, default='max',
                        help='Pooling method')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate.')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs.')
    parser.add_argument('--use_all', action='store_true', default=False,
                        help='Use all data.')
    parser.add_argument('--data', type=str, default='en',
                        help='Dataset.')
    parser.add_argument('--train_size', type=int, default=3000,
                        help='Train size.')
    parser.add_argument('--test_size', type=int, default=1000,
                        help='Test size.')
    parser.add_argument('--val_portion', type=float, default=0.1, 
                        help='Validation portion.')
    parser.add_argument('--hidden_dim', type=int, default=25,
                        help='Hidden dimension.')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--early_stopping', type=int, default=100,
                        help='Tolerance for early stopping (# of epochs).')
    parser.add_argument('--min_frequency', type=int, default=5,
                        help='Minimum frequency of words.')
    return vars(parser.parse_args())
