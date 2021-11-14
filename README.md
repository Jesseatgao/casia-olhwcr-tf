## CASIA OLHWCR-TF

Online handwriting Chinese character recognition using Tensorflow 2, Keras & Flask , based on CASIA's GB2312 level-1 dataset

### SCREENSHOTS

![screenshots](https://raw.githubusercontent.com/Jesseatgao/casia-olhwcr-tf/conv_with_blstm/docs/recognition.gif)

### Installation

* from within source directory locally

    `pip install .`

### USAGE

>**:envelope:**\
> The model restored from the pre-saved checkpoint (located in 'olccr/recognition/conf/checkpoint/weights.hdf5')
> is not fully trained, and its hyperparameters are not tuned. The app above
> is just for demonstrating the idea. Further training and experimentation should be done.

#### Running recognition App

From the Bash/CMD shell execute the following command, then visit http://127.0.0.1:5000/:

```shell
(ENV)$ olccr
```

#### Re-training model

* Preparing raw data

Run the following command to unzip and patch the raw data:

```shell
(ENV)$ olccr_prepare
```

* Making dataset

To generate the training and validation dataset, simply run:

```shell
(ENV)$ olccr_preprocess -t -v
```

* Training model

Run or re-run after interruption the following command to train or resume training the network, respectively

```shell
(ENV)$ olccr_train -V 1
```

#### Installing trained weights 

With default setup, for example, the checkpointed weights should be saved as `olccr/data/ckpts/weights.hdf5`. To apply
the latest weights, copy it to the App's recognition configuration directory `olccr/recognition/conf/checkpoint/` 
replacing the same name file, then restart the App.

### REFERENCES

* https://github.com/taosir/cnn_handwritten_chinese_recognition
* https://github.com/Leimi/drawingboard.js
* https://github.com/michael-zhu-sh/CASIA/blob/master/OLHWDB/OLHWDB1.cpp
* https://zhuanlan.zhihu.com/p/101513445
* http://www.nlpr.ia.ac.cn/databases/handwriting/Online_database.html
* http://www.herongyang.com/GB2312/GB2312-to-Unicode-Map-Level-1-Characters.html

### DONATE

If you like the project, please support it by donation
[![PayPal donate button](https://img.shields.io/badge/paypal-donate-yellow.svg)](
    https://www.paypal.com/cgi-bin/webscr?cmd=_xclick&business=changxigao@gmail.com&item_name=Support%20me%20by%20donating&currency_code=USD)