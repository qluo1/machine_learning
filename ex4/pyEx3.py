from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import numpy
import scipy
from scipy.io import loadmat as loadmat

data = loadmat("ex4data1.mat")
para = loadmat("ex4weights.mat")

X = data['X']
y = data['y']
Theta1 = para['Theta1']
Theta2 = para['Theta2']

n = FeedForwardNetwork()
inLayer, hiddenLayer, outLayer = SigmoidLayer(400),SigmoidLayer(25), SigmoidLayer(10)

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer,hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer,outLayer)

n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

n.sortModules()
print n


ds = SupervisedDataSet(400,10)

for idx,v in enumerate(X):
    ds.addSample(v,y[idx] == y[idx][0])

print len(ds)
trainer = BackpropTrainer(n, ds)
from datetime import datetime
start = datetime.now()

for i in range(1000):
    print trainer.train()

#print trainer.trainUntilConvergence()
print n.params.shape

print datetime.now() - start
