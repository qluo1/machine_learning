from scipy.io import loadmat as loadmat
from sklearn.naive_bayes import GaussianNB

data = loadmat("ex8data1.mat")

X = data['X']
Xval = data['Xval']
yval = data['yval']

gnd = GaussianNB()

pred = gnd.fit(Xval,yval)
y_exp = pred.predict(X)


data = loadmat("ex8data2.mat")

X = data['X']
Xval = data['Xval']
yval = data['yval']

gnd = GaussianNB()
pred = gnd.fit(Xval,yval)
y_exp = pred.predict(X)

print X[y_exp == 1,]
