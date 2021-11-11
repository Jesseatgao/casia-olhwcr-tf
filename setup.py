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
        'numpy==1.19.5',
        'Flask==1.1.2',
        'Jinja2==2.11.3; python_version=="3.6"',
        'Pillow==8.3.1',
        'opencv-python==4.5.3.56',
        'tensorflow==2.6.0',
        'tensorflow-estimator==2.6.0',
        'tensorboard==2.6.0',
        'keras==2.6.0'
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
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
