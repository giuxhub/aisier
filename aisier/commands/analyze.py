import argparse

from aisier.aisier import Aisier


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='aisier view', description='View the model structure and training statistics.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='Path of the project.')
    parser.add_argument('-n', '--num_of_features', action='store', dest='num_of_features', type=int, default=10,
                        help='The top <n> number of features to analyze.')

    args = parser.parse_args(argv)
    return args


def command_analyze(argv):
    args = parse_args(argv)
    aisier = Aisier(args.path)

    err = aisier.analyze(args.num_of_features)
    if err is not None:
        print('error while loading project: %s', err)
        quit()
