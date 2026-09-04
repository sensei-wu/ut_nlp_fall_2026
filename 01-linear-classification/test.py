from algorithms.binary_lr import lr
import numpy as np

X = np.array([[1,0],[1,1],[0,1],[0,1]])
y = np.array([1,1,0,0])

w, b = lr(X, y, 100, 1)

print(w, b)