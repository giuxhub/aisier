import argparse

from aisier.aisier import Aisier


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='aisier view', description='View the model structure and training statistics.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='Path of the project.')

    args = parser.parse_args(argv)
    return args


def command_view(argv):
    args = parse_args(argv)
    aisier = Aisier(args.path)

    err = aisier.view()
    if err is not None:
        print('error while loading project: %s', err)
        quit()
