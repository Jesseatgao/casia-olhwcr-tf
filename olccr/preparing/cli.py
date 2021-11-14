from os.path import abspath, dirname, join as join_path
from os import makedirs
from argparse import ArgumentParser

from .preparation import unzip_train_pot, unzip_val_pot


pardir = abspath(dirname(dirname(__file__)))
data_dir = join_path(pardir, 'data')

default_working_dir = data_dir


def _arg_parser():
    parser = ArgumentParser()

    parser.add_argument('-d', '--working-dir', default=default_working_dir, dest='working_dir',
                        help='working directory in which to save all the related files [default: olccr/data]')
    parser.add_argument('-t', '--train-pot-dir', default='train_pot', dest='train_pot_dir',
                        help='directory containing the training POT files; relative to `working_dir`. '
                             'if not given, it will default to `train_pot`')
    parser.add_argument('-v', '--validation-pot-dir', default='val_pot', dest='val_pot_dir',
                        help='directory containing the validation POT files; relative to `working_dir`. '
                             'if not given, it will default to `val_pot`')

    return parser


def main():
    args = _arg_parser().parse_args()

    working_dir = args.working_dir
    train_pot_dir = abspath(join_path(working_dir, args.train_pot_dir))
    val_pot_dir = abspath(join_path(working_dir, args.val_pot_dir))

    # create the specified dirs if not present
    makedirs(train_pot_dir, exist_ok=True)
    makedirs(val_pot_dir, exist_ok=True)

    unzip_train_pot(train_pot_dir)
    unzip_val_pot(val_pot_dir)
