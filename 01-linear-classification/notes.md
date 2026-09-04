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

## 04.09.2026

- Perceptron is a type of mistake bounded learning. It can be seen as a special case of SGD. It is a linear model that learns a weight from data, but it is not a probabilistic model

- LR is a probabilistic, discriminative model. It is also a linear model that learns a weight from data. But it learns by maximizing the log likelihood of correct labels given the data (ie, probability of a given class conditional on corresponding data samples)

- NB is a probabilistic, generative model. NB also relies on maximizing the log likelihood, but in contrast to LR which maximizes the conditional likelihood of labels given data, NB maximizes the joint likelihood of data and classes. NB has a closed form and thus learns by counting rather than search and then uses Bayes' rule to infer the class.

### iris dataset

Refactored to use perceptron and LR. This required adjusting the labels before calling each of them.

### Python

- Sigmoid function overflow occured in the initial implementation. I had to adjust it using a standard trick to avoid overflowing 
when z is large negative, since $e^{-z}$ then exceeds float64 range; branch on the sign so the exponent is always negative.

### Key learnings

The two models disagree. Perceptron $[−0.19, −0.60, 0.76, 0.32]$ vs LR $[−0.11, −0.53, 0.78, 0.33]$, normalized. Both separate the data, but they're different hyperplanes. The perceptron stops at the first separator it finds, while LR keeps optimizing past separation (until convergence condition).