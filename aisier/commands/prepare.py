import os
import glob
import json
import argparse

from tqdm import tqdm
from aisier.aisier import Aisier


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='aisier prepare',
                                     description='Encode one or more files to vectors and create or update a csv dataset for training.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model', help='The path containing the model definition.')
    parser.add_argument('path', help='Path to a folder of files.')

    parser.add_argument('-e', '--extension', dest='extension', default='*.*',
                        help='This flag determines which files extensions are going to be selected.')

    args = parser.parse_args(argv)
    return args


def folder_name(path):
    return os.path.basename(os.path.dirname(path))


def generate_labels(path, path_to_save):
    idx = 0
    labels = dict()
    subfolders = glob.glob(os.path.join(path, "*"))

    for subfolder in subfolders:
        labels[os.path.basename(subfolder)] = idx
        idx += 1

    with open(path_to_save, 'w') as f:
        json.dump(labels, f)

    return labels


def command_prepare(argv):
    args = parse_args(argv)
    if not os.path.exists(args.path):
        print('{} does not exist.'.format(args.path))
        quit()

    aisier = Aisier(args.model)
    err = aisier.load()
    if err is not None:
        print('error while loading project')
        quit()

    if os.path.exists(aisier.label_path):
        with open(aisier.label_path, 'r') as fp:
            labels = json.load(fp)
            print('loading labels...')
    else:
        labels = generate_labels(args.path, aisier.label_path)
        print('generating labels...')

    inputs = []
    if os.path.isdir(args.path):
        in_files = []
        for subfolder in glob.glob(os.path.join(args.path, "*")):
            print('enumerating {}...'.format(subfolder))
            in_filter = os.path.join(subfolder, args.extension)
            in_sub = glob.glob(in_filter)
            in_files.extend(in_sub)

        for filepath in in_files:
            if os.path.isfile(filepath):
                inputs.append((labels[folder_name(filepath)], filepath))
    else:
        print('you need to provide a data folder')
        quit()

    # x is the filename and y the folder name
    for (y, x) in tqdm(inputs, total=len(inputs), desc='Processing files: '):
        err = aisier.prepare(x, y)
        if err is not None:
            print(err)
            quit()
