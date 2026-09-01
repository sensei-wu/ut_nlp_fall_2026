from sklearn.datasets import load_iris
from perceptron import perceptron
import numpy as np

def load_data():
    print('loading...')
    data = load_iris()
    print(f"Loaded iris dataset with features: {data.feature_names} and targets {data.target_names}")

    samples, labels = data.data, data.target
    print(f"Sample data {samples[0]}, sample label: {labels[0]}")
    print(f"Unique labels found: {np.unique(labels)}")
    mask = (labels == 0) | (labels == 1)
    X, y = samples[mask], labels[mask]
    y_signed = 2 * y - 1
    print(f"Unique labels adjusted: {np.unique(y_signed)}") # confirm
    return X, y_signed, data.feature_names

X, y, feature_names = load_data()

w = perceptron(X, y)

print(f"Found weight: {w}")

for name, weight in zip(feature_names, w):
    print(f"{name:20s} {weight:+.2f}")

print(X[y == -1].mean(axis=0))
print(X[y == 1].mean(axis=0))