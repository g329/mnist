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


class MnistModel(chainer.Chain):
    def __init__(self):

        super(MnistModel, self).__init__(
                l1=L.Linear(784, 100),
                l2=L.Linear(100, 100),
                l3=L.Linear(100, 10),
        )

    def __call__(self, x, t, train=True):

        x = chainer.Variable(x)
        t = chainer.Variable(t)

        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.accuracy(h, t)


model = MnistModel()
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(100):
    model.zerograds()
    loss, acc = model(train_data, train_label)
    loss.backward()
    print "epoch %3d : %.4f" % (epoch, acc.data)
    optimizer.update()

model.zerograds()
acc = model(test_data, test_label, train=False)
print "test acc : ", acc.data
