# Notes for week 1:

## 01.09.2026

### Setup

I decided for a project structure that reflects course structure -- ut_nlp_fall_2026/week#-module. I use conda for environment management. Project uses python 3.11. I plan to add dependencies in `requirements.txt`

### Perceptron

This is among the simplest of all classifier algorithms. Despite the simplicity, it offers convergence under appropriate conditions:

- finite dataset
- data is linearly separable with a margin $\gamma$
- $||x||$ is bounded

Perceptron starts with a random or zero weight vector and iteratively updates the weights when mistakes occur.

### iris dataset

In-order to test the algorithm, I decided to use a standard dataset from scikit-learn. [Iris dataset](https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html) is a really small dataset that serves the purpose. It contains 150 labeled samples of Iris flower, each with 4 features. I used the dataset to develop and debug perceptron.

### Key learnings

- python allows `else` block for `for` loops. 
- perceptron requires labels in {-1, 1}. However the conditional in python `w @ X[i] < 0` returns `True` or `False`. I was comparing the boolean result of the evaluation to -1 which always returned `False`. My first implementation thus resulted in a bug which resulted in absurd learned weights. Eyeballing the printed results was a good idea