import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def initialize_theta(d):
    theta = np.ones([d, 1])
    return theta


def convert_df_to_np_array(x):
    x_np = x.to_numpy()
    return x_np



def main():
    #d = int(input("Enter the no. of dimensions for the model: "))
    theta = initialize_theta(6)
    #print(theta.shape)
    sim_data_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    y = sim_data_df["disease_score"]
    y_true = y.to_numpy()
    y_actual = y_true.reshape(60, 1)
    #print(y_actual)
    #print(y_actual.shape)
    x = sim_data_df.drop(["disease_score", "disease_score_fluct"], axis=1)
    X = convert_df_to_np_array(x)
    x_main = np.c_[np.ones(X.shape[0]), X]
    #print(x_main.shape)
    x_transpose = x_main.T
    #print(x_transpose.shape)
    iterations = 10000
    alpha = 0.000001
    lamda = 10
    y_pred = np.dot(x_main, theta)
    sse_initial = (np.sum((y_pred - y_actual) ** 2))/2
    #percent_sse = 6
    for iter in range(iterations):
    #while percent_sse > 0.05:
        error = y_pred - y_actual
        gradient = np.dot(x_transpose, error)
        theta = theta * (1 - alpha * lamda) - alpha * gradient
        y_pred = np.dot(x_main, theta)
        #print(y_pred)
        sse = (np.sum((y_pred - y_actual) ** 2))/2
        percent_sse = (abs(sse_initial - sse)/sse_initial) * 100
        sse_initial = sse
        if percent_sse <= 0.001:
            print("This is the SSE.")
            print(sse)
            break
        print("This is the SSE.")
        print(sse)
    print("Final SSE.")
    y_pred = np.dot(x_main, theta)
    print((np.sum((y_pred - y_actual) ** 2))/2)



    print(theta)
    print(theta.shape)
    #sse =
    print(y_pred)
    print(r2_score(y_actual, y_pred))









if __name__ == '__main__':
    main()




