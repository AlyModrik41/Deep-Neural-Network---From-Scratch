import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def tanh(v):
    return np.tanh(v)

def forward_step(sample, weights, bias, activation_name, b_use):
    acc_array = [sample]
    curr_sample = sample
    for i in range(len(weights)):
        w = weights[i]
        v = np.dot(w, curr_sample)
        if b_use:
            v += bias[i]
        if activation_name == 'sigmoid':
            curr_out = sigmoid(v)
        elif activation_name == 'tanh':
            curr_out = tanh(v)
        acc_array.append(curr_out)
        curr_sample = curr_out
    return acc_array

def compute_errors(weights, acc_array, one_hot_vector, activation_name):
    error = []
    output = acc_array[-1]
    if activation_name == 'sigmoid':
        derivative = output * (1 - output)
    elif activation_name == 'tanh':
        derivative = 1 - output * output
    error.append((one_hot_vector - output) * derivative)
    error_inst = error[-1]
    for i in range(1, len(weights)):
        output = acc_array[-i - 1]
        if activation_name == 'sigmoid':
            derivative = output * (1 - output)
        elif activation_name == 'tanh':
            derivative = 1 - output * output
        error_inst = np.dot(weights[-i].T, error_inst) * derivative
        error.append(error_inst)
    return error

def update_weights(weights, acc_array, errors, eta, bias, b_use):
    for i in range(1, len(weights) + 1):
        weights[-i] = weights[-i] + eta * np.dot(errors[i-1], acc_array[-(i+1)].T)
        if b_use:
            bias[-i] = bias[-i] + eta * errors[i-1]
    return weights, bias

def confusion_matrix(y_true, y_pred, classes):
    matrix = [[0] * classes for i in range(classes)]
    for y_values in range(len(y_pred)):
        pred = y_pred[y_values]
        true = y_true[y_values]
        matrix[true][pred] += 1
    return matrix

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return (correct / len(y_true)) * 100


# ── DATA ──────────────────────────────────────────────
df = pd.read_csv('df_encoded.csv')
X = df.drop(columns=['Species'])
y = df['Species']
class_names = ['Adelie', 'Chinstrap', 'Gentoo']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)