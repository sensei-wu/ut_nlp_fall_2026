import numpy as np

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        e = np.exp(x)
        return e / (1 + e)

def lr(X,y, epochs, alpha):
    w = np.zeros(X.shape[1])
    b = 0
    for t in range(epochs):
        w_prev = w.copy()
        for n in range(len(X)):
            z = b + w @ X[n]
            p = sigmoid(z)
            err = y[n] - p
            w += alpha * err * X[n]
            b += alpha * err

        if t > 0:
            w_hat = w / np.linalg.norm(w)
            w_prev_hat = w_prev / np.linalg.norm(w_prev)
            if np.linalg.norm(w_hat - w_prev_hat) < 1e-3:
                print(f"LR converged after {t + 1} epochs")
                break
        print(f"After {t+1} epoch, weight norm = {np.linalg.norm(w)}, weight = {w}")
    return w, b            
