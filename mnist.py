# -*- coding:utf-8 -*-
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

mnist = fetch_mldata('MNIST original', data_home='.')
mnist.data = mnist.data.astype(np.float32) * 1.0 / 255.0
mnist.target = mnist.target.astype(np.int32)
train_data, test_data, train_label, test_label = train_test_split(mnist.data, mnist.target, test_size=10000,
                                                                  random_state=222)

print "data shape ", mnist.data.dtype, mnist.data.shape
print "label shape ", mnist.target.dtype, mnist.target.shape

