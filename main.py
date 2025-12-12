import random

import numpy
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


class Catfish:
    def __init__(self):
        self.weights = np.zeros((7, 7, 2))
        self.learning_rate = 0.1
        self.coop = 2
        self.init_weights()

    def init_weights(self):
        mx = self.weights.shape[0]
        my = self.weights.shape[1]
        for i in range(mx):
            for j in range(my):
                self.weights[i, j] = (i, j)

    def fit(self, X, epochs: int = 10):
        mx = self.weights.shape[0]
        my = self.weights.shape[1]
        for epoch in range(1, epochs + 1):
            np.random.shuffle(X)

            for x in X:
                i = self.l2_per_element(self.weights, x).argmin()
                i, j = (i % mx, i // my)
                # h = np.full(
                #     (mx, my),
                #     fill_value=lambda i,j: )

    def l2_per_element(self, a, b):
        mx = self.weights.shape[0]
        my = self.weights.shape[1]
        res = numpy.zeros((mx, my))
        for x in range(mx):
            for y in range(my):
                res[x, y] = self.l2(a[x, y], b)
        return res

    def l2(self, a, b):
        return np.linalg.norm(a - b)


X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
som = Catfish()
r = som.l2_per_element(som.weights, X[0])
print(r)
print(r.argmin())
print(r[])
