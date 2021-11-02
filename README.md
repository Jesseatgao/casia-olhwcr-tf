##CASIA OLHWCR-TF
Online handwriting Chinese character recognition using Tensorflow 2, Keras & Flask , based on CASIA's GB2312 level-1 dataset

###SCREENSHOTS
![screenshots](https://raw.githubusercontent.com/Jesseatgao/casia-olhwcr-tf/conv_with_blstm/docs/recognition.gif)

###USAGE
>**:envelope:**\
> The model restored from the pre-saved checkpoint (located in 'app/recognition/conf/checkpoint/weights.hdf5')
> is not fully trained, and its hyperparameters are not tuned. The app above
> is just for demonstrating the idea. Further training and experimentation should be done.

####Preprocessing

####Training

###REFERENCES
* https://github.com/taosir/cnn_handwritten_chinese_recognition
* https://github.com/Leimi/drawingboard.js
* https://github.com/michael-zhu-sh/CASIA/blob/master/OLHWDB/OLHWDB1.cpp
* https://zhuanlan.zhihu.com/p/101513445
* http://www.nlpr.ia.ac.cn/databases/handwriting/Online_database.html
* http://www.herongyang.com/GB2312/GB2312-to-Unicode-Map-Level-1-Characters.html

###DONATE
If you like the project, please support it by donation
[![PayPal donate button](https://img.shields.io/badge/paypal-donate-yellow.svg)](
    https://www.paypal.com/cgi-bin/webscr?cmd=_xclick&business=changxigao@gmail.com&item_name=Support%20me%20by%20donating&currency_code=USD)