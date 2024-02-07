import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def initialize_theta(f):
    theta = np.ones([f, 1])
    theta_T = theta.T
    return theta_T


def sigmoid(s):
    r = 1 / (1 + np.exp(-1*s))
    return r


def main():
    f = int(input("How many features(x) does your Data Have? "))
    theta = initialize_theta(f)
    sim_df = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
    x_df = sim_df.drop(["disease_status"], axis=1)
    y_df = sim_df["disease_status"]
    x = x_df.to_numpy().reshape(500, 4)
    #x_main = x.T
    y_true = y_df.to_numpy().reshape(500, 1)
    #x_main = np.c_[np.ones(x.shape[0]), x]
    y = np.dot(x, theta.T)
    pred_val = sigmoid(y)
    # print(h)
    alpha = 0.000001
    iterations = 100000
    for iter in range(iterations):
        theta = theta + ((alpha * (y_true - pred_val).T) @ x)
        y = np.dot(x, theta.T)
        pred_val = sigmoid(y)
    print(theta)
    y_predict = np.dot(x, theta.T)
    y_list = list(y_predict)
    y_list_pred = []
    for i in y_list:
        if i > 0.5:
            y_list_pred.append(1)
        else:
            y_list_pred.append(0)
    y_final = np.array(y_list_pred).reshape(500, 1)
    print(y_final)
    acc_score = accuracy_score(y_true, y_final)
    print(acc_score)


    #print(y_predict)


if __name__ == "__main__":
    main()



