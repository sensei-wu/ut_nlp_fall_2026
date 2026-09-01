import numpy as np

def perceptron(X, Y):
    w = np.zeros(X.shape[1])
    NUM_EPOCHS = 10
    for t in range(NUM_EPOCHS):
        err_cnt = 0
        for i in range(len(X)):
            #print(f"Sample {i}, Data: {X[i]}, Label: {Y[i]}")
            if w @ X[i] < 0:
                y_pred = -1
            else:
                y_pred = 1
            if y_pred != Y[i]:
                w += Y[i] * X[i]
                err_cnt += 1
        if err_cnt == 0:
            print(f"Perceptron converged in {t+1} epochs")
            break
    else:
        print(f"no convergence in {NUM_EPOCHS} epochs")
    return w