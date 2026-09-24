"""
This module implements a simplest possible, sentiment based XOR in a canonical pytorch way.
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
    layer1 = torch.nn.Linear(2, 2)
    layer2 = torch.nn.Linear(2, 2)

    print(f"layer1 weight {layer1.weight}, layer2 weight {layer2.weight}")
    print(f"layer1 bias {layer1.bias}, layer2 bias {layer2.bias}")

    out = layer1(X)
    print(type(out), out.shape, out.grad_fn)

    h = torch.relu(out)
    logits = layer2(h)
    print(h.shape, logits.shape)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)
    print(f"Loss before training and backpropagation (which is expected to be roughly -ln(0.5)) {loss.item()}")

    # real training
    lr = 0.1

    opt = torch.optim.SGD(list(layer1.parameters()) + list(layer2.parameters()), lr = lr)
    for n in range(1000):
        h = torch.relu(layer1(X))
        logits = layer2(h)

        opt.zero_grad()
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if n % 100 == 0:
            print(f"after {n} iterations, loss = {loss.item()}")

    with torch.no_grad():
        print("after transformation by hidden:")
        h = torch.relu(layer1(X))
        logits = layer2(h)
        print(h)
        print(f"predicted: {logits.argmax(dim=1)}, expected: {y}")



