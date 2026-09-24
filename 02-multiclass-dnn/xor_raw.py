"""
This module implements a sentiment based XOR from first principles: raw tensors, manual gradient updates, no nn.Module.

Written to demonstrate what PyTorch's abstractions actually do. Usual approach uses
nn.Module and torch.optim; see xor_module.py for the same network built that way.
"""

import torch

def featurize(sentence: str) -> list[int]:
    words = sentence.lower().split()
    return [int("good" in words), int("not" in words)]

if __name__ == "__main__":
    data = [("good", 1), ("bad", 0), ("not bad", 1), ("not good", 0)]
    X_feats = [featurize(x) for x, _ in data]
    y_labels = [label for _, label in data]
    print(f"Data: {X_feats},  label: {y_labels}")

    # pytorch starts here
    X = torch.tensor(X_feats, dtype=torch.float32)
    y = torch.tensor(y_labels)
    print(f"X = {X}, y = {y}")

    torch.manual_seed(0)

    # initial pass just for print-debug
    W1 = torch.randn(2, 2, requires_grad=True)
    b1 = torch.zeros(2, requires_grad=True)
    W2 = torch.randn(2, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)

    h = torch.relu( X @ W1 + b1)
    logits = h @ W2 + b2
    print(h.shape, logits.shape)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)
    print(f"Loss before training and backpropagation (which is expected to be roughly -ln(0.5)) {loss.item()}")

    # real training
    lr = 0.1
    for n in range(1000):
        h = torch.relu( X @ W1 + b1)
        #h = X @ W1 + b1. uncomment this line and comment above line to see how ReLu helps learning by introducing non linearity
        logits = h @ W2 + b2
        loss = loss_fn(logits, y)
        loss.backward()

        with torch.no_grad():
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad

            W1.grad.zero_()
            b1.grad.zero_()
            W2.grad.zero_()
            b2.grad.zero_()

        if n % 100 == 0:
            print(f"after {n} iterations, loss = {loss.item()}")

    with torch.no_grad():
        print("after transformation by hidden:")
        h = torch.relu( X @ W1 + b1)
        #h = X @ W1 + b1. uncomment this line and comment above line to see how ReLu helps learning by introducing non linearity
        logits = h @ W2 + b2
        print(h)
        print(f"predicted: {logits.argmax(dim=1)}, expected: {y}")



