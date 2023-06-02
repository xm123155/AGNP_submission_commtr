import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--train', type=bool, default=False, help='train or test')
parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train.')
parser.add_argument('--hiddenEnc', type=int, default=32, help='Number of units in hidden layer of Encoder.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_str', type=str, default='AGNP', help='type of dataset.')
parser.add_argument('--max_norm', type=int, default=5, help='max norm for gradient clipping')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--connectedness_matrix', type=str, default='data/connectedness.pkl',
                    help='connectedness matrix position')

args = parser.parse_args()
