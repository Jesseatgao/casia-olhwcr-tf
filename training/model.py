import os
import glob
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers


def load_data_from_disk(files):
    for fn in files:
        with open(fn, 'rb') as fd:
            data = pickle.load(fd)

        for imgs, label in data:
            imgs = tf.reshape(imgs, [-1, 32, 32, 1])

            # do the normalization on the fly to save disk space (Preprocessing should have been done in real life!)
            imgs = tf.cast(imgs, tf.float32)  # tf.uint8 -> tf.float32
            imgs = tf.divide(imgs, 255)

            yield imgs, label


def build_input_pipeline(dataset_dir):
    data_file_pat = os.path.normpath(os.path.join(dataset_dir, r"**/*.pkl"))
    data_files = glob.glob(data_file_pat, recursive=True)
    dataset = tf.data.Dataset.from_generator(
        load_data_from_disk,
        output_signature=(
            tf.TensorSpec(shape=(None, 32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)),
        args=(data_files,)
    )

    return dataset


# FIXME: hyperparameters
def _build_model(training=True):
    inputs = keras.Input(shape=(None, 32, 32, 1), dtype=tf.float32, ragged=True, name="input_layer")

    # Conv
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', name="conv_layer_1")
    timedist_conv1 = layers.TimeDistributed(conv1)(inputs)
    maxpool1 = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_layer_1")
    timedist_maxpool1 = layers.TimeDistributed(maxpool1)(timedist_conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', name="conv_layer_2")
    timedist_conv2 = layers.TimeDistributed(conv2)(timedist_maxpool1)
    maxpool2 = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_layer_2")
    timedist_maxpool2 = layers.TimeDistributed(maxpool2)(timedist_conv2)

    # projection
    conv3 = layers.Conv2D(16, 1, activation='relu', padding='same', name="conv_layer_3")
    timedist_conv3 = layers.TimeDistributed(conv3)(timedist_maxpool2)

    timedist_flatten1 = layers.TimeDistributed(layers.Flatten())(timedist_conv3)

    # LSTM with long-term features
    blstm1 = layers.Bidirectional(layers.LSTM(16, dropout=0.3, name="blstm_layer_1"))(timedist_flatten1,
                                                                                      training=training)

    flatten1 = layers.Flatten(name="flatten_layer_1")(blstm1)

    # LSTM with short-term features
    timedist_flatten2 = layers.TimeDistributed(layers.Flatten())(inputs)

    blstm2 = layers.Bidirectional(layers.LSTM(16, dropout=0.3, name="blstm_layer_2"))(timedist_flatten2,
                                                                                      training=training)

    flatten2 = layers.Flatten(name="flatten_layer_2")(blstm2)

    # max Conv over the pic series from the first stroke to the very last big pic
    globalmaxpool1 = layers.GlobalMaxPool1D()(timedist_flatten1)

    # flatten3 = layers.Flatten(name="flatten_layer_3")(globalmaxpool1)

    concat1 = layers.Concatenate(name="concat_layer_1")([flatten1, flatten2, globalmaxpool1])

    dense1 = layers.Dense(1024, activation='relu', name="dense_layer_1")(concat1)
    dropout1 = layers.Dropout(0.3, name="dropout_layer_1")(dense1, training=training)

    outputs = layers.Dense(3755, name="output_layer")(dropout1)

    model = keras.Model(inputs=inputs, outputs=outputs)

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

    model.compile(optimizer=optimizers.RMSprop(),
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
                                                     patience=2, verbose=0)

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

    from preprocess import make_dataset_gb2312_level1


    def _arg_parser():
        parser = ArgumentParser()

        parser.add_argument('-d', '--working-dir', required=True, dest='working_dir',
                            help='working directory in which to save all the related files')
        parser.add_argument('-t', '--train-pot-dir', default='train_pot', dest='train_pot_dir',
                            help='directory containing the training POT files; relative to `working_dir`')
        parser.add_argument('-v', '--validation-pot-dir', default='val_pot', dest='val_pot_dir',
                            help='directory containing the validation POT files; relative to `working_dir`')
        parser.add_argument('-D', '--dataset-dir', default='dataset', dest='dataset_dir',
                            help='directory in which to save the generated training and test dataset in pickle format; '
                                 'relative to `working_dir`')
        parser.add_argument('-C', '--checkpoint-dir', default='ckpts', dest='checkpoint_dir',
                            help='directory in which to save the model checkpoints; relative to `working_dir`')
        parser.add_argument('-T', '--tensorboard-dir', default='tb_logs', dest='tensorboard_dir',
                            help='directory in which to save the tensorboard logs; relative to `working_dir`')
        parser.add_argument('-R', '--backup-dir', default='backup_n_restore', dest='backup_dir',
                            help='directory in which to save the BackupAndRestore logs; relative to `working_dir`')
        parser.add_argument('-E', '--epochs', default=15000, dest='epochs', type=int,
                            help='epochs to run for training, though early stopping may occur')
        parser.add_argument('-B', '--batch-size', default=32, dest='batch_size', type=int,
                            help='mini batch size for training the model')

        return parser

    args = _arg_parser().parse_args()

    # FIXME: hyperparameters
    working_dir = args.working_dir
    train_pot_dir = os.path.join(working_dir, args.train_pot_dir)
    val_pot_dir = os.path.join(working_dir, args.val_pot_dir)
    dataset_dir = os.path.join(working_dir, args.dataset_dir)
    checkpoint_dir = os.path.join(working_dir, args.checkpoint_dir)
    tensorboard_dir = os.path.join(working_dir, args.tensorboard_dir)
    backup_dir = os.path.join(working_dir, args.backup_dir)

    epochs = args.epochs
    batch_size = args.batch_size

    train_dataset_dir, val_dataset_dir, _ = make_dataset_gb2312_level1(dataset_dir,
                                                                       train_pot_dir=train_pot_dir,
                                                                       val_pot_dir=val_pot_dir)

    train_data = build_input_pipeline(train_dataset_dir)
    train_data = train_data.shuffle(buffer_size=512)
    train_data = train_data.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    train_data_iter = iter(train_data)

    val_data = build_input_pipeline(val_dataset_dir)
    val_data = val_data.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    val_data_iter = iter(val_data)

    model, callbacks = make_train_model(checkpoint_dir=checkpoint_dir,
                                        tensorboard_dir=tensorboard_dir,
                                        backup_dir=backup_dir)

    # collect the state of the iterators automatically
    model._train_data_iter = train_data_iter
    model._val_data_iter = val_data_iter

    model.summary()

    model.fit(train_data_iter, epochs=epochs, validation_data=val_data_iter, callbacks=callbacks)

