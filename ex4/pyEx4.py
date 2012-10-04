from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet

net = buildNetwork(400,25,10,bias=True,hiddenclass=SigmoidLayer)

import numpy
import scipy
from scipy.io import loadmat as loadmat

data = loadmat("ex4data1.mat")
X = data['X']
y = data['y']

ds = SupervisedDataSet(400,10)

for idx,v in enumerate(X):
    ds.addSample(v,y[idx] == y[idx][0])

print len(ds),net
trainer = BackpropTrainer(net, ds)

for i in range(100):
    print trainer.train()

print net.params,net.params.shape
#print trainer.trainUntilConvergence()


