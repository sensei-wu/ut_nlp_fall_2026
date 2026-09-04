from sklearn.datasets import load_iris
from algorithms.binary_perceptron import perceptron
from algorithms.binary_lr import lr
import numpy as np

def load_data():
    print('loading...')
    data = load_iris()
    print(f"Loaded iris dataset with features: {data.feature_names} and targets {data.target_names}")

    samples, labels = data.data, data.target
    print(f"Sample data {samples[0]}, sample label: {labels[0]}")
    print(f"Unique labels found: {np.unique(labels)}")
    return samples, labels, data.feature_names

def prepare_perceptron(X, y):
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    y_signed = 2 * y - 1
    print(f"Unique labels adjusted: {np.unique(y_signed)}") # confirm
    return X, y_signed

def prepare_lr(X, y):
    mask = (y == 0) | (y == 1)
    X, y = X[mask], y[mask]
    print(f"Unique labels adjusted: {np.unique(y)}") # confirm
    return X, y

data, labels, feature_names = load_data()

X, y = prepare_perceptron(data, labels)

print(f".......finding weights using binary perceptron.......")

w = perceptron(X, y)

print(f"Found weight: {w}")

for name, weight in zip(feature_names, w):
    print(f"{name:20s} {weight:+.2f}")

#print(X[y == -1].mean(axis=0))
#print(X[y == 1].mean(axis=0))

epochs_lr = 100
alpha_lr = 1

X, y = prepare_lr(data, labels)

print(f".......finding weights using LR.......")

w_lr = lr(X, y, epochs_lr, alpha_lr)

print(f"Found weight: {w_lr}")