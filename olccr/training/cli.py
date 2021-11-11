from os.path import abspath, dirname, join as join_path
from argparse import ArgumentParser

from .model import make_train_model, build_input_pipeline


pardir = abspath(dirname(dirname(__file__)))
data_dir = join_path(pardir, 'data')

default_working_dir = data_dir


def _arg_parser():
    parser = ArgumentParser()

    parser.add_argument('-d', '--working-dir', default=default_working_dir, dest='working_dir',
                        help='working directory in which to save all the related files')
    parser.add_argument('-t', '--train-dataset-dir', default='dataset/training', dest='train_dataset_dir',
                        help='directory containing the training data files in pickle format; '
                             'relative to `working_dir`')
    parser.add_argument('-v', '--validation-dataset-dir', default='dataset/validation', dest='val_dataset_dir',
                        help='directory containing the validation data files in pickle format; '
                             'relative to `working_dir`')

    parser.add_argument('-C', '--checkpoint-dir', default='ckpts', dest='checkpoint_dir',
                        help='directory in which to save the model checkpoints; relative to `working_dir`')
    parser.add_argument('-T', '--tensorboard-dir', default='tb_logs', dest='tensorboard_dir',
                        help='directory in which to save the tensorboard logs; relative to `working_dir`')
    parser.add_argument('-R', '--backup-dir', default='backup_n_restore', dest='backup_dir',
                        help='directory in which to save the BackupAndRestore logs; relative to `working_dir`')
    parser.add_argument('-P', '--patience', default=20, dest='patience', type=int,
                        help='number of epochs with no improvement after which training will be stopped')
    parser.add_argument('-E', '--epochs', default=1000, dest='epochs', type=int,
                        help='epochs to run for training, though early stopping may occur')
    parser.add_argument('-B', '--batch-size', default=32, dest='batch_size', type=int,
                        help='mini batch size for training the model')
    parser.add_argument('-S', '--shuffle-buffer-size', default=1200, dest='shuffle_size', type=int,
                        help='buffer size for shuffling the training elements using two-level shuffling')
    parser.add_argument('-V', '--verbose', default=2, dest='verbose', type=int, choices=[0, 1, 2],
                        help='verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')

    return parser


def main():
    args = _arg_parser().parse_args()

    working_dir = args.working_dir

    train_dataset_dir = join_path(working_dir, args.train_dataset_dir)
    val_dataset_dir = join_path(working_dir, args.val_dataset_dir)

    checkpoint_dir = join_path(working_dir, args.checkpoint_dir)
    tensorboard_dir = join_path(working_dir, args.tensorboard_dir)
    backup_dir = join_path(working_dir, args.backup_dir)

    patience = args.patience
    epochs = args.epochs
    batch_size = args.batch_size
    shuffle_size = args.shuffle_size
    verbose = args.verbose

    # build the training and validation data pipeline
    train_dataset = build_input_pipeline(train_dataset_dir, shuffle=True, elems_shuffle_size=shuffle_size,
                                         batch_size=batch_size)
    val_dataset = build_input_pipeline(val_dataset_dir, shuffle=False, batch_size=batch_size)

    model, callbacks = make_train_model(checkpoint_dir=checkpoint_dir,
                                        tensorboard_dir=tensorboard_dir,
                                        backup_dir=backup_dir,
                                        patience=patience)

    # collect the state of the datasets automatically
    model._train_dataset = train_dataset
    model._val_dataset = val_dataset

    model.summary(line_length=150)

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, verbose=verbose)
