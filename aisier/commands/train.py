import os
import argparse

from aisier.aisier import Aisier


def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError('{} not in range [0.0 - 1.0]'.format(x))
    return x


def validate_args(args):
    p_train = 1 - args.test - args.validation
    if p_train < 0:
        print('validation and test proportion are bigger than 1')
        quit()
    elif p_train < 0.5:
        print('using less than 50% of the dataset for training')
        quit()


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='aisier train', description='Start the training phase of a model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', help='The path containing the model definition.')

    parser.add_argument('-d', '--dataset', action='store', dest='dataset',
                        help='Path of the dataset to use. Leave empty to reuse previously generated subsets.')
    parser.add_argument('-t', '--test', action='store', dest='test', type=probability, default=0.15,
                        help='Proportion of the test set in the (0,1] interval.')
    parser.add_argument('-v', '--validation', action='store', dest='validation', type=probability, default=0.15,
                        help='Proportion of the validation set in the (0,1] interval.')

    args = parser.parse_args(argv)
    validate_args(args)

    return args


def command_train(argv):
    args = parse_args(argv)
    aisier = Aisier(args.path)
    err = aisier.load()
    if err is not None:
        print('error while loading project: {}'.format(err))
        quit()

    if aisier.dataset.exists():
        aisier.dataset.load_dataset()
    elif args.dataset is not None:
        aisier.dataset.build_dataset(args.dataset, args.test, args.validation)
    else:
        print('no test/train/validation subsets found in {} and no dataset provided'.format(args.path))
        quit()

    aisier.train()
