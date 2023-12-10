import torch
import numpy as np  
import matplotlib.pyplot as plt

# Regression Evaluation

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def corr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def evaluate_regression(y_true, y_pred):
    return {'mae': mae(y_true, y_pred),
            'corr': corr(y_true, y_pred)}

def plot_regression(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=1)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    print(evaluate_regression(y_true, y_pred))
    plot_regression(y_true, y_pred, 'test')