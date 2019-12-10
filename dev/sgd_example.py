'''
SGD classifier example adapted from https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.linear_model.SGDClassifier.html

The purpose of this is to confirm the formatting for input vectors, expected by SGD'''
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])

# examine vector shape, to infer data formatting expected by SGD
print("X.shape", X.shape)
print("Y.shape", Y.shape)

clf = linear_model.SGDClassifier()
clf.fit(X, Y)

SGDClassifier(alpha=0.0001, class_weight=None, eta0=0.0,
        fit_intercept=True, learning_rate='optimal', loss='hinge',
        max_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
        shuffle=False, verbose=0, warm_start=False)
print(clf.predict([[-0.8, -1]]))
