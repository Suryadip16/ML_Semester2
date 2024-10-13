import numpy as np
from sklearn.datasets import fetch_california_housing
X, Y = fetch_california_housing(return_X_y=True, as_frame=True)
x = X.to_numpy()
y_prime = Y.to_numpy().reshape(1, 20639)


def initialize_theta(d):
    theta = np.ones([d, 1])
    return theta


# def convert_df_to_np_array(x):
#     x_np = x.to_numpy()
#     return x_np


def main():
    d = int(input("Enter the no. of dimensions for the model: "))
    theta = initialize_theta(d)
    #sim_data_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    #y = sim_data_df["disease_score"]
    #y_true = y.to_numpy()
    y_actual = y
    #x = sim_data_df.drop(["disease_score", "disease_score_fluct"], axis=1)
    #X = convert_df_to_np_array(x)
    x_main = np.c_[np.ones(X.shape[0]), X]
    x_transpose = x_main.T
    iterations = 1000000
    alpha = 0.000001
    y_pred = np.dot(x_main, theta)
    idx = np.random.randint(0, 60)
    sse_initial = ((y_pred[idx, 0] - y_actual[idx, 0]) ** 2)/2

    for iter in range(iterations):
        error = y_pred[idx, 0] - y_actual[idx, 0]
        gradient = np.dot(x_transpose[:, idx], error).reshape(6, 1)
        theta = theta - alpha * gradient
        y_pred = np.dot(x_main, theta)
        idx = np.random.randint(0, 60)

        sse = (np.sum((y_pred[idx, 0] - y_actual[idx, 0]) ** 2))/2
        percent_sse = (abs(sse_initial - sse)/sse_initial) * 100

        sse_initial = sse
        if percent_sse <= 0.1:
            print("This is the SSE.")
            print(sse)
            print("Percent SSE: ", percent_sse )
            break
        print("This is the SSE.")
        print(sse)
    print(theta)
    print(theta.shape)
    print(y_pred)


if __name__ == '__main__':
    main()

