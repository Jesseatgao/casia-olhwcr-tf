from os.path import abspath, dirname, join as join_path
from argparse import ArgumentParser

from .preprocess import make_dataset_gb2312_level1


pardir = abspath(dirname(dirname(__file__)))
data_dir = join_path(pardir, 'data')

default_working_dir = data_dir


def _arg_parser():
    parser = ArgumentParser()

    parser.add_argument('-d', '--working-dir', default=default_working_dir, dest='working_dir',
                        help='working directory in which to save all the related files')
    parser.add_argument('-t', '--train-pot-dir', default=None, dest='train_pot_dir', nargs='?', const='train_pot',
                        help='directory containing the training POT files; relative to `working_dir`. '
                             'if not given, it will default to `None`; with no argument followed, '
                             'it will consume a value of `train_pot`')
    parser.add_argument('-v', '--validation-pot-dir', default=None, dest='val_pot_dir', nargs='?', const='val_pot',
                        help='directory containing the validation POT files; relative to `working_dir`. '
                             'if not given, it will default to `None`; with no argument followed, '
                             'it will consume a value of `val_pot`')
    parser.add_argument('-f', '--font-file', default=None, dest='font_file',
                        help='true type font file used to generate the standard Chinese character images, '
                             'e.g. /absolute/path/to/simhei.ttf')
    parser.add_argument('-B', '--pot-batch', default=10, dest='pot_batch', type=int,
                        help='number of POT files combined to produce one of the training datasets')
    parser.add_argument('-S', '--image-size', default=32, dest='img_size', type=int,
                        help='size of the training/validation sample image and/or the Chinese character font image')
    parser.add_argument('-D', '--dataset-dir', default='dataset', dest='dataset_dir',
                        help='directory in which to save the generated training and test dataset in pickle format; '
                             'relative to `working_dir`')

    return parser


def main():
    args = _arg_parser().parse_args()

    working_dir = args.working_dir
    dataset_dir = join_path(working_dir, args.dataset_dir)

    train_pot_dir = join_path(working_dir, args.train_pot_dir) if args.train_pot_dir else args.train_pot_dir
    val_pot_dir = join_path(working_dir, args.val_pot_dir) if args.val_pot_dir else args.val_pot_dir
    font_file = args.font_file
    pot_batch = args.pot_batch
    img_size = args.img_size

    train_dataset_dir, val_dataset_dir, dict_dataset_dir = make_dataset_gb2312_level1(dataset_dir,
                                                                                      train_pot_dir=train_pot_dir,
                                                                                      val_pot_dir=val_pot_dir,
                                                                                      pot_batch=pot_batch,
                                                                                      font=font_file,
                                                                                      img_size=img_size)
