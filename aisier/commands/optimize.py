import os
import argparse

from aisier.aisier import Aisier


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='aisier optimize-dataset',
                                     description='Remove duplicates from the dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', help='Path of the model folder.')
    parser.add_argument('-f', '--filename', dest='filename', action='store', type=str, default='dataset.csv',
                        help='Name of the dataset file to optimize.')

    args = parser.parse_args(argv)
    return args


def command_optimize_dataset(argv):
    args = parse_args(argv)
    path = os.path.abspath(args.path)

    if not os.path.exists(path):
        print('model folder {} does not exist'.format(path))
        quit()

    aisier = Aisier(path)
    err = aisier.optimize(args.filename)
    if err is not None:
        print(err)
        quit()
