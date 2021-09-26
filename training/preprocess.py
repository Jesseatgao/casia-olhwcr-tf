import os
from io import BytesIO
import glob
from distutils.dir_util import mkpath
import pickle
from struct import unpack

import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont


def get_chinese_from_gbk(rows_cols_pairs):
    """

    Args:
        rows_cols_pairs: [([rows], [cols]), ([rows], [cols]),]

    Returns:

    """
    for rows, cols in rows_cols_pairs:
        for row in rows:
            for col in cols:
                han = (row << 8 | col).to_bytes(2, 'big').decode(encoding='gbk')  # Chinese character
                yield han


def build_gb2312_level1_charset():
    """

    Returns:

    References:
        https://www.herongyang.com/GB2312/GB2312-to-Unicode-Map-Level-1-Characters.html

    """
    rows_seg1 = range(0xB0, 0xD7)
    cols_seg1 = range(0xA1, 0xFF)

    rows_seg2 = range(0xD7, 0xD8)
    cols_seg2 = range(0xA1, 0xFA)

    level1_pairs = [(rows_seg1, cols_seg1), (rows_seg2, cols_seg2)]
    hans = [han for han in get_chinese_from_gbk(level1_pairs)]

    return hans


POT_SAMPLE_HEAD_SIZE = 8  # 2B (sample size) + 4B (tag code) + 2B (stroke number), see POT file format
POT_SAMPLE_POINT_SIZE = 4  # 2B + 2B ( coordinates (x, y) )


def _consume_sample_head(fd):
    header = fd.read(POT_SAMPLE_HEAD_SIZE)
    if len(header) < POT_SAMPLE_HEAD_SIZE:
        return None, None, None

    sample_size, tag_code, stroke_num = unpack('<H4sH', header)
    if tag_code[0]:  # non-ASCII
        codepoint = tag_code[1] << 8 | tag_code[0]  # little endian ordering for multi-byte character
        nbytes = 2
    else:
        codepoint = tag_code[0] << 8 | tag_code[1]  # mock 'big endian' for ASCII character (only 1-byte valid)
        nbytes = 1
    char = codepoint.to_bytes(nbytes, 'big').decode(encoding='gbk')  # Chinese character OR ASCII symbol

    return sample_size, char, stroke_num


def _consume_sample_strokes(fd, stroke_num):
    strokes = []

    for i in range(stroke_num):
        stroke = []
        while True:
            point = fd.read(POT_SAMPLE_POINT_SIZE)
            x, y = unpack('<hh', point)
            if not (x == -1 and y == 0):  # not the stroke end
                stroke.append((x, y))
            else:
                strokes.append(stroke)

                break

    return strokes


def _draw_img_seq_from_strokes(strokes):
    thickness = 7
    padding = 7
    min_x, min_y, max_x, max_y = 32767, 32767, 0, 0

    # find the bounding box
    for stroke in strokes:
        for x, y in stroke:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    box_width = max_x - min_x + padding * 2
    box_height = max_y - min_y + padding * 2
    x_shift, y_shift = -min_x + padding, -min_y + padding

    images = []
    curves = []

    # generate a sequence of images following the strokes, incrementally
    for stroke in strokes:
        if len(stroke) < 2:  # noise
            continue

        pts = np.array(stroke)                 # points forming a polygonal curve
        pts = np.add(pts, (x_shift, y_shift))  # convert to relative coordinates
        curves.append(pts)

        canvas = np.zeros((box_height, box_width), dtype=np.uint8)  # black background
        img = cv.polylines(canvas, [pts], False, color=(255, 255, 255), thickness=thickness, lineType=cv.LINE_AA)
        images.append(img)

    canvas = np.zeros((box_height, box_width), dtype=np.uint8)
    big_pic = cv.polylines(canvas, curves, False, color=(255, 255, 255), thickness=thickness, lineType=cv.LINE_AA)

    return images, big_pic


def get_imgs_from_casia_pot(fn):
    """Get the labelled images of Chinese characters from a CASIA pot file.

    Args:
        fn:

    Returns:

    """
    with open(fn, mode="rb") as fd:
        while True:
            _, char, stroke_num = _consume_sample_head(fd)
            if char is None:
                break

            strokes = _consume_sample_strokes(fd, stroke_num)
            imgs, big_pic = _draw_img_seq_from_strokes(strokes)

            yield imgs, big_pic, char

            # consume the character end tag
            fd.read(POT_SAMPLE_POINT_SIZE)


def get_chars_from_casia_pot(fn):
    """

    Args:
        fn:

    Returns:

    References:
        http://www.nlpr.ia.ac.cn/databases/handwriting/Online_database.html
    """
    with open(fn, mode="rb") as fd:
        while True:
            sample_size, char, _ = _consume_sample_head(fd)
            if char is None:
                break

            yield char

            # skip the rest of the current sample
            fd.seek(sample_size - POT_SAMPLE_HEAD_SIZE, os.SEEK_CUR)


def build_charset_from_train_pots(dirname, charset_size=None):
    """

    Args:
        dirname:
        charset_size (int): The predetermined number of classes of characters in the training data files.

    Returns:

    """
    path_pattern = os.path.normpath(os.path.join(dirname, r"**/*.pot"))
    if charset_size is None:
        charset = {char for fn in glob.glob(path_pattern, recursive=True) for char in get_chars_from_casia_pot(fn)}
    else:
        charset = set()
        for fn in glob.glob(path_pattern, recursive=True):
            chars = {char for char in get_chars_from_casia_pot(fn)}
            charset |= chars
            if len(charset) == charset_size:
                break

    chars = list(charset)

    return chars


def get_hans_png_bytes(hans, font, size=32, bg_color='black', txt_color='white'):
    """

    Args:
        hans (str):
        font (str):
        size (int):
        bg_color (str):
        txt_color (str):

    Returns:

    """
    image = Image.new('L', (size, size), bg_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font, size)
    draw.text((0, 0), hans, fill=txt_color, font=font)

    with BytesIO() as buf:
        image.save(buf, format='PNG')
        png = buf.getvalue()

    return png


def cv_resize_img_retaining_AR(img, size=(32, 32), interpolation=None):
    """Resize the image while keeping the aspect ratio of the original shape.

    Args:
        img (:obj:`np.ndarray`): The input image to resize, assuming which has a black background.
        size (2-tuple of int): The width and height of the desired output image.
        interpolation (int): The interpolation method, e.g. ``cv.INTER_AREA``.

    Returns:
        ``np.ndarray``: The resized image.

    References:
        https://stackoverflow.com/questions/44650888
    """
    h, w = img.shape[:2]  # height, width
    sw = h if h > w else w  # embedding square width

    if interpolation is None:
        interpolation = cv.INTER_AREA if sw > (size[0] + size[1]) // 2 else cv.INTER_CUBIC

    if h == w:
        return cv.resize(img, size, interpolation)

    x_pos = (sw - w) // 2
    y_pos = (sw - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((sw, sw), dtype=img.dtype)  # black background
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img
    else:
        ch = img.shape[2]  # channel number
        mask = np.zeros((sw, sw, ch), dtype=img.dtype)  # black background
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img

    return cv.resize(mask, size, interpolation)


cv_resize = cv_resize_img_retaining_AR


def make_dataset_gb2312_level1(dataset_dir, train_pot_dir=None, val_pot_dir=None, pot_batch=10,
                               font=None, img_size=32, normalization=False, id_hans_dicts=True):
    # subdirs relative to `dataset_dir`
    TRAIN_DATASET_SUBDIR = "training"  # 'train_data.pkl', or 'train_data-001.pkl', 'train_data-002.pkl', ...
    VAL_DATASET_SUBDIR = "validation"  # 'val_data.pkl', or 'val_data-001.pkl', 'val_data-002.pkl', ...
    DICT_DATASET_SBUDIR = "dicts"      # 'id2hans.pkl', 'id2png.pkl'

    train_dataset_dir, val_dataset_dir, dict_dataset_dir = None, None, None

    def dump(data, to_fn):
        """write the pickled `data` to the file named `to_fn`."""
        # create the (missing) dirs following the path
        path2fn = os.path.dirname(to_fn)
        if path2fn and not os.path.isdir(path2fn):
            mkpath(path2fn)  # FIXME: exceptions may be raised

        with open(to_fn, 'wb') as data_fd:
            pickle.dump(data, data_fd, protocol=3)

    def hans_id_dict():
        """craft the lookup table of Chinese characters and their assigned class ID numbers."""
        hans = build_gb2312_level1_charset()
        hans_to_id = dict(zip(sorted(hans), range(len(hans))))
        id_to_hans = {clsid: hans for hans, clsid in hans_to_id.items()}

        return hans_to_id, id_to_hans

    hans2id, id2hans = hans_id_dict()

    def process_image(img, new_size, norm):
        # scale the image to (new_size, new_size) pixels
        img = cv_resize(img, size=(new_size, new_size))

        # normalize the image
        if norm:
            img = img.astype("float32") / 255.0

        return img

    def make_data(to_fn, from_dir):
        """generate the training and test dataset."""
        pot_pattern = os.path.normpath(os.path.join(from_dir, r"**/*.pot"))
        pots = sorted(glob.glob(pot_pattern, recursive=True))  # make sure that the dataset creation is idempotent
        if not pots:
            print("No POT file(s) found in '{}'!".format(from_dir))
            return

        fb, ext = os.path.splitext(to_fn)
        n = 1
        for i in range(0, len(pots), pot_batch):
            fs = pots[i:i+pot_batch]

            mid = "-{:03}".format(n) if pot_batch < len(pots) else ""
            data_fn = "{}{}{}".format(fb, mid, ext)

            data = [([process_image(img, img_size, normalization) for img in imgs],
                     process_image(big_pic, img_size, normalization), hans2id[char])
                    for fn in fs for imgs, big_pic, char in get_imgs_from_casia_pot(fn)]
            dump(data, data_fn)

            n += 1

    if train_pot_dir:
        train_dataset_dir = os.path.abspath(os.path.join(dataset_dir, TRAIN_DATASET_SUBDIR))

        train_data_pn = os.path.join(train_dataset_dir, 'train_data.pkl')  # train dataset template pathname
        make_data(train_data_pn, train_pot_dir)

    if val_pot_dir:
        val_dataset_dir = os.path.abspath(os.path.join(dataset_dir, VAL_DATASET_SUBDIR))

        val_data_pn = os.path.join(val_dataset_dir, 'val_data.pkl')  # validation dataset template pathname
        make_data(val_data_pn, val_pot_dir)

    if id_hans_dicts:
        dict_dataset_dir = os.path.abspath(os.path.join(dataset_dir, DICT_DATASET_SBUDIR))

        # dump the id-to-hans_text dict
        id2hans_pathname = os.path.join(dict_dataset_dir, 'id2hans.pkl')
        dump(id2hans, id2hans_pathname)

        # make the id-to-printed_Chinese_character_image dict
        if font:
            # map ID number to PNG image
            id2png = {clsid: get_hans_png_bytes(hans, font, size=img_size) for hans, clsid in hans2id.items()}
            id2png_pathname = os.path.join(dict_dataset_dir, 'id2png.pkl')
            dump(id2png, id2png_pathname)

    return train_dataset_dir, val_dataset_dir, dict_dataset_dir


if __name__ == "__main__":
    from argparse import ArgumentParser

    def _arg_parser():
        parser = ArgumentParser()

        parser.add_argument('-d', '--working-dir', required=True, dest='working_dir',
                            help='working directory in which to save all the related files')
        parser.add_argument('-t', '--train-pot-dir', default='train_pot', dest='train_pot_dir', nargs='?', const=None,
                            help='directory containing the training POT files; relative to `working_dir`. '
                                 'if not given, it will default to `train_pot`; with no argument followed, '
                                 'it will consume a value of `None`')
        parser.add_argument('-v', '--validation-pot-dir', default='val_pot', dest='val_pot_dir', nargs='?', const=None,
                            help='directory containing the validation POT files; relative to `working_dir`. '
                                 'if not given, it will default to `val_pot`; with no argument followed, '
                                 'it will consume a value of `None`')
        parser.add_argument('-f', '--font-file', default=None, dest='font_file',
                            help='true type font file used to generate the standard Chinese character images, '
                                 'e.g. /absolute/path/to/simhei.ttf')
        parser.add_argument('-B', '--pot-batch', default=10, dest='pot_batch', type=int,
                            help='number of POT files combined to produce one of the training datasets')
        parser.add_argument('-D', '--dataset-dir', default='dataset', dest='dataset_dir',
                            help='directory in which to save the generated training and test dataset in pickle format; '
                                 'relative to `working_dir`')

        return parser

    args = _arg_parser().parse_args()

    working_dir = args.working_dir
    dataset_dir = os.path.join(working_dir, args.dataset_dir)

    train_pot_dir = os.path.join(working_dir, args.train_pot_dir) if args.train_pot_dir else args.train_pot_dir
    val_pot_dir = os.path.join(working_dir, args.val_pot_dir) if args.val_pot_dir else args.val_pot_dir
    font_file = args.font_file
    pot_batch = args.pot_batch

    train_dataset_dir, val_dataset_dir, dict_dataset_dir = make_dataset_gb2312_level1(dataset_dir,
                                                                                      train_pot_dir=train_pot_dir,
                                                                                      val_pot_dir=val_pot_dir,
                                                                                      pot_batch=pot_batch,
                                                                                      font=font_file)
