import os
from base64 import b64decode, b64encode
from io import BytesIO
import pickle

from flask import Blueprint, render_template, json, jsonify, request
from PIL import Image
import numpy as np
import tensorflow as tf

from app.training.model import build_model


recognition_bp = Blueprint('recognition_bp', __name__, template_folder='templates', static_folder='static',
                           static_url_path='recognition/static')

here = os.path.dirname(os.path.abspath(__file__))
checkpoint = os.path.normpath(os.path.join(here, 'conf/checkpoint/weights.hdf5'))
id2png_pkl = os.path.normpath(os.path.join(here, 'conf/dicts/id2png.pkl'))


@recognition_bp.route('/')
def index():
    return render_template('recognition/drawing.html')


model = build_model(training=False)
model.load_weights(checkpoint)

with open(id2png_pkl, 'rb') as fd:
    id2png = pickle.load(fd)


def _convert_image(image, output_shape):
    decoded = b64decode(image)
    with BytesIO(decoded) as bfd:
        with Image.open(bfd) as img:
            converted = img.convert('L')
            converted = np.asarray(converted, dtype='float32') / 255.0
            converted = converted.reshape(output_shape)

    return converted


def _inputs_gen(strokes, bigPic):
    imgs = [_convert_image(stroke, 1024) for stroke in strokes]
    big_pic = _convert_image(bigPic, (32, 32, 1))

    label = -1  # unknown class ID

    yield imgs, big_pic, label


def _build_inputs(strokes, bigPic):
    inputs = tf.data.Dataset.from_generator(
        _inputs_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 1024), dtype=tf.float32),
            tf.TensorSpec(shape=(32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)),
        args=(strokes, bigPic)
    )
    inputs = inputs.map(lambda imgs, big_pic, label: ((imgs, big_pic), label))
    inputs = inputs.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=1))

    return inputs


def _build_response(inputs):
    k = 3
    prediction = model.predict(inputs)
    prediction = tf.nn.softmax(prediction)
    probs, indices = tf.nn.top_k(prediction, k=k)

    probs, indices = probs.numpy(), indices.numpy()

    resp = {}
    for i in range(k):
        pred_image = 'pred{}_image'.format(i + 1)
        resp[pred_image] = 'data:image/png;base64,' + b64encode(id2png[indices[0, i]]).decode()

        pred_accuracy = 'pred{}_accuracy'.format(i + 1)
        resp[pred_accuracy] = '{:.2%}'.format(probs[0, i])

    return jsonify(resp)


@recognition_bp.route('/recognize', methods=['POST'])
def recognize():
    strokes = json.loads(request.form.get('strokes'))
    bigPic = request.form.get('bigPic')

    return _build_response(_build_inputs(strokes, bigPic))
