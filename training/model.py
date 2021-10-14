import os
import glob
import pickle
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers


def load_data_from_disk(files, shuffle=True, files_shuffle_size=None):
    dataset = tf.data.Dataset.from_tensor_slices(files)

    if shuffle:
        buffer_size = len(files) if files_shuffle_size is None else files_shuffle_size
        dataset = dataset.shuffle(buffer_size)

    for fn in dataset.as_numpy_iterator():
        with open(fn, 'rb') as fd:
            data = pickle.load(fd)

        for imgs, big_pic, label in data:
            imgs = tf.reshape(imgs, [-1, 1024])

            # do the normalization on the fly to save disk space (Preprocessing should have been done in real life!)
            imgs = tf.cast(imgs, tf.float32)  # tf.uint8 -> tf.float32
            imgs = tf.divide(imgs, 255)

            big_pic = tf.reshape(big_pic, [32, 32, 1])
            big_pic = tf.cast(big_pic, tf.float32)
            big_pic = tf.divide(big_pic, 255)

            yield imgs, big_pic, label


def build_input_pipeline(dataset_dir, shuffle=True, files_shuffle_size=None, elems_shuffle_size=1200, batch_size=32, prefetch_size=1):
    data_file_pat = os.path.normpath(os.path.join(dataset_dir, r"**/*.pkl"))
    data_files = sorted(glob.glob(data_file_pat, recursive=True))  # always return the files in the same order

    data_gen = partial(load_data_from_disk, files_shuffle_size=files_shuffle_size)
    dataset = tf.data.Dataset.from_generator(
        data_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 1024), dtype=tf.float32),
            tf.TensorSpec(shape=(32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)),
        args=(data_files, shuffle)
    )
    dataset = dataset.map(lambda imgs, big_pic, label: ((imgs, big_pic), label))

    if shuffle:
        dataset = dataset.shuffle(elems_shuffle_size)

    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    dataset = dataset.prefetch(prefetch_size)

    return dataset


# FIXME: hyperparameters
def _build_model(training=True):
    inputs_lstm = keras.Input(shape=(None, 1024), dtype=tf.float32, ragged=True, name="inputs_lstm")
    inputs_conv = keras.Input(shape=(32, 32, 1), dtype=tf.float32, name="inputs_conv")

    # Conv
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', name="conv_layer_1")(inputs_conv)
    conv1_1 = layers.Conv2D(32, 3, activation='relu', padding='same', name="conv_layer_1_1")(conv1)

    maxpool1 = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_layer_1")(conv1_1)
    dropout1 = layers.Dropout(0.2, name="dropout_layer_1")(maxpool1, training=training)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', name="conv_layer_2")(dropout1)
    conv2_1 = layers.Conv2D(64, 3, activation='relu', padding='same', name="conv_layer_2_1")(conv2)

    maxpool2 = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_layer_2")(conv2_1)
    dropout2 = layers.Dropout(0.2, name="dropout_layer_2")(maxpool2, training=training)

    # flatten1 = layers.Flatten()(dropout2)

    # projection
    conv3 = layers.Conv2D(16, 1, activation='relu', padding='same', name="conv_layer_3")(dropout2)

    flatten1 = layers.Flatten()(conv3)

    # lstm
    blstm1 = layers.Bidirectional(layers.LSTM(64, dropout=0.2, name="blstm_layer_1"))(inputs_lstm, training=training)

    blstm1_dropout = layers.Dropout(0.2, name="dropout_blstm1")(blstm1, training=training)

    flatten2 = layers.Flatten(name="flatten_layer_1")(blstm1_dropout)

    concat1 = layers.Concatenate(name="concat_layer_1")([flatten1, flatten2])

    dense1 = layers.Dense(1024, activation='relu', name="dense_layer_1")(concat1)
    dense1_dropout = layers.Dropout(0.3, name="dense1_dropout")(dense1, training=training)

    outputs = layers.Dense(3755, name="output_layer")(dense1_dropout)

    model = keras.Model(inputs=[inputs_lstm, inputs_conv], outputs=outputs)

    return model


def make_train_model(checkpoint_dir, tensorboard_dir, backup_dir):

    def _setup_log_dirs():
        for d in [checkpoint_dir, tensorboard_dir, backup_dir]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    _setup_log_dirs()

    def _get_saved_checkpoint():
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, r"*.hdf5"))
        if ckpt_files:
            latest_checkpoint = max(ckpt_files, key=os.path.getctime)
            return latest_checkpoint

    latest_checkpoint = _get_saved_checkpoint()

    # setup the training (compiled) model
    model = _build_model(training=True)

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    def _build_callbacks(model):

        ckpt_file = latest_checkpoint if latest_checkpoint else os.path.join(checkpoint_dir, "weights.hdf5")
        ckpt_cb = keras.callbacks.ModelCheckpoint(filepath=ckpt_file, save_weights_only=True,
                                                  monitor='val_accuracy', mode='max',
                                                  save_best_only=True, verbose=0)

        tensorboard_cb = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                                                     embeddings_freq=0, update_freq='epoch')

        earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2,
                                                     patience=5, verbose=0)

        backup_cb = keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir)

        # let the model automatically track these callbacks
        model._ckpt_cb = ckpt_cb
        model._tensorboard_cb = tensorboard_cb
        model._earlystop_cb = earlystop_cb

        callbacks = [ckpt_cb, tensorboard_cb, earlystop_cb, backup_cb]

        return callbacks

    callbacks = _build_callbacks(model)

    return model, callbacks


if __name__ == "__main__":
    from argparse import ArgumentParser

    def _arg_parser():
        parser = ArgumentParser()

        parser.add_argument('-d', '--working-dir', required=True, dest='working_dir',
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
        parser.add_argument('-E', '--epochs', default=100, dest='epochs', type=int,
                            help='epochs to run for training, though early stopping may occur')
        parser.add_argument('-B', '--batch-size', default=32, dest='batch_size', type=int,
                            help='mini batch size for training the model')
        parser.add_argument('-S', '--shuffle-buffer-size', default=1200, dest='shuffle_size', type=int,
                            help='buffer size for shuffling the training elements using two-level shuffling')
        parser.add_argument('-V', '--verbose', default=2, dest='verbose',  type=int, choices=[0, 1, 2],
                            help='verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')

        return parser

    args = _arg_parser().parse_args()

    # FIXME: hyperparameters
    working_dir = args.working_dir

    train_dataset_dir = os.path.join(working_dir, args.train_dataset_dir)
    val_dataset_dir = os.path.join(working_dir, args.val_dataset_dir)

    checkpoint_dir = os.path.join(working_dir, args.checkpoint_dir)
    tensorboard_dir = os.path.join(working_dir, args.tensorboard_dir)
    backup_dir = os.path.join(working_dir, args.backup_dir)

    epochs = args.epochs
    batch_size = args.batch_size
    shuffle_size = args.shuffle_size
    verbose = args.verbose

    # build the training and validation data pipeline
    train_dataset = build_input_pipeline(train_dataset_dir, shuffle=True, elems_shuffle_size=shuffle_size, batch_size=batch_size)
    val_dataset = build_input_pipeline(val_dataset_dir, shuffle=False, batch_size=batch_size)

    model, callbacks = make_train_model(checkpoint_dir=checkpoint_dir,
                                        tensorboard_dir=tensorboard_dir,
                                        backup_dir=backup_dir)

    # collect the state of the datasets automatically
    model._train_dataset = train_dataset
    model._val_dataset = val_dataset

    model.summary(line_length=150)

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, verbose=verbose)

