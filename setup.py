from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'olccr', 'VERSION'), 'r', 'utf-8') as fd:
    version = fd.read().strip()

with codecs.open(os.path.join(here, 'README.md'), 'r', 'utf-8') as fd:
    long_description = fd.read()

# Extends the Setuptools `clean` command
with open(os.path.join(here, 'third_parties', 'setupext_janitor', 'janitor.py')) as setupext_janitor:
    exec(setupext_janitor.read())

# try:
#     from setupext_janitor import janitor
#     CleanCommand = janitor.CleanCommand
# except ImportError:
#     CleanCommand = None

cmd_classes = {}
if CleanCommand is not None:
    cmd_classes['clean'] = CleanCommand

setup(
    name='casia-olhwcr-tf',
    version=version,
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    include_package_data=True,
    package_data={
        'olccr': ['data/raw/*', 'VERSION'],
        'olccr.recognition': ['conf/checkpoint/*', 'conf/dicts/*', 'static/css/*', 'static/images/*', 'static/js/*',
                              'templates/recognition/*']
    },
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.19.5; python_version<"3.10"',
        'numpy==1.24.*; python_version>="3.10" and python_version<"3.12"',
        'numpy; python_version>="3.12"',
        'Flask==1.1.2; python_version<"3.10"',
        'Flask',
        'Jinja2==2.11.3; python_version=="3.6"',
        'jinja2<3.1.0; python_version<"3.10"',
        'Pillow==8.3.1; python_version<"3.10"',
        'Pillow',
        'opencv-python==4.5.3.56; python_version<"3.10"',
        'opencv-python',
        'tensorflow==2.6.0; python_version<"3.10"',
        'tensorflow==2.15.*; python_version>="3.10" and python_version<"3.12"',
        'tensorflow; python_version>="3.12"',
        'tensorflow-estimator==2.6.0; python_version<"3.10"',
        'tensorflow-estimator; python_version<"3.12"',
        'tensorboard==2.6.0; python_version<"3.10"',
        'tensorboard',
        'keras==2.6.0; python_version<"3.10"',
        'keras<3.0.0; python_version>="3.10" and python_version<"3.12"',
        'tf-keras~=2.16; python_version>="3.12"',
        'grpcio<=1.74.0; python_version<"3.10"',
        'itsdangerous==2.0.1; python_version<"3.10"',
        'werkzeug==2.0.3; python_version<"3.10"',
        'protobuf<=3.20.3; python_version<"3.10"'
    ],
    setup_requires=[],
    cmdclass=cmd_classes,
    entry_points={
        'console_scripts': [
            'olccr_prepare = olccr.preparing.cli:main',
            'olccr_preprocess = olccr.preprocess.cli:main',
            'olccr_train = olccr.training.cli:main',
            'olccr = olccr.cli:main'
        ],
        'distutils.commands': [
            'clean = CleanCommand'
        ]
    },
    url='https://github.com/Jesseatgao/casia-olhwcr-tf',
    license='MIT License',
    author='Jesse Gao',
    author_email='changxigao@gmail.com',
    description='Online handwriting Chinese character recognition using Tensorflow 2, Keras & Flask , based on CASIA\'s GB2312 level-1 dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Developers',
        'Environment :: Console',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
