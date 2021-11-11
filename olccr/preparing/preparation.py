import glob
from os.path import abspath, dirname, join as join_path
from shutil import copyfile
import zipfile
import tempfile


pardir = abspath(dirname(dirname(__file__)))
data_dir = join_path(pardir, 'data')


def unzip_pot(zipf, dst_dir):
    with zipfile.ZipFile(zipf) as zip_fd:
        zip_fd.extractall(path=dst_dir)


def unzip_split_pot(zipfs, dst_dir):
    def _concat_file_pieces(fnames, tmp_fd):
        for fn in fnames:
            with open(fn, 'rb') as fd:
                tmp_fd.write(fd.read())

        tmp_fd.seek(0)

    with tempfile.TemporaryFile(mode='w+b', suffix='.zip', dir=dst_dir) as tmp_fd:
        _concat_file_pieces(zipfs, tmp_fd)
        unzip_pot(tmp_fd, dst_dir)


def unzip_train_pot(dst_dir):
    zipfs = sorted(glob.glob(join_path(data_dir, 'raw', 'OLHWDB1.1trn_pot.zip.*')))
    unzip_split_pot(zipfs, dst_dir)

    patch_1043_c = join_path(data_dir, 'raw', '1043-c.pot.patched')
    orig_1043_c = join_path(dst_dir, '1043-c.pot')
    copyfile(patch_1043_c, orig_1043_c)


def unzip_val_pot(dst_dir):
    zipf = join_path(data_dir, 'raw', 'OLHWDB1.1tst_pot.zip')
    unzip_pot(zipf, dst_dir)
