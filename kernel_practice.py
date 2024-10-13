import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def transform(X, y):
    dim = X.shape
    x_transform = np.zeros((dim[0], dim[1] + 1))
    for i in range(dim[0]):
        x_transform[i, 0] = X[i, 0] ** 2
        x_transform[i, 1] = np.sqrt(2) * X[i, 0] * X[i, 1]
        x_transform[i, 2] = X[i, 1] ** 2
    print(x_transform)
    x_final = np.concatenate((x_transform, y), axis=1)
    return x_final


def transform_no_label(X):
    dim = X.shape
    x_transform = np.zeros((dim[0], dim[1] + 1))
    for i in range(dim[0]):
        x_transform[i, 0] = X[i, 0] ** 2
        x_transform[i, 1] = np.sqrt(2) * X[i, 0] * X[i, 1]
        x_transform[i, 2] = X[i, 1] ** 2

    return x_transform


def polynomial_kernel(x1, x2):
    res = x1[0] ** 2 * x2[0] ** 2 + 2 * x1[0] * x2[0] * x1[1] * x2[1] + x1[1] ** 2 * x2[1] ** 2
    return res


def rbf_kernel(X, y, sigma):
    dim = X.shape
    x_rbf_kerneled = np.zeros((dim[0], dim[1] - 1))
    for i in range(dim[0]):
        x_rbf_kerneled[i, 0] = np.exp(-1 * (((X[i, 0] - X[i, 1]) ** 2) / (2 * sigma ** 2)))
    x_rbf_final = np.concatenate((x_rbf_kerneled, y), axis=1)
    return x_rbf_final




def main():
    df = pd.read_csv('kernel_practice data.csv')

    x_df = df.drop(["Label"], axis=1)
    label_df = df["Label"]
    label_np = label_df.to_numpy().reshape(16, 1)
    print(label_np)
    x_np = x_df.to_numpy()
    x_3d = transform(x_np, label_np)
    plt.scatter(df["x1"], df["x2"], c=df['Label'])
    plt.show()
    x = x_3d[:, 0]
    y = x_3d[:, 1]
    z = x_3d[:, 2]
    color = x_3d[:, 3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=color)
    plt.show()
    x1 = [3, 6]
    x2 = [10, 10]
    dot_pd = 0
    for i in range(len(x1)):
        prd = x1[i] * x2[i]
        dot_pd = dot_pd + prd
    print(f"Dot product in 2D: {dot_pd}")
    x1_np = np.array([x1, x2])
    x1_final = transform_no_label(x1_np).T
    col1 = x1_final[:, 0]
    col2 = x1_final[:, 1]
    dot_prd_3d = np.dot(col1, col2)
    print(f"Dot Product in 3D: {dot_prd_3d}")
    kernel_val = polynomial_kernel(x1, x2)
    print(f"Value with Polynomial Kernel: {kernel_val}")
    rbf_data = pd.read_csv("rbf_kernel_data.csv")
    rbf_df = rbf_data.drop(["Label"], axis=1)
    label_rbf_df = rbf_data["Label"]
    label_rbf_np = label_rbf_df.to_numpy().reshape(15, 1)
    print(label_rbf_np)
    x_rbf_np = rbf_df.to_numpy()
    s = 0.0000001
    x_rbf_final = rbf_kernel(x_rbf_np, label_rbf_np, s)
    x_rbf_final_dim = x_rbf_final.shape
    x_axis = np.arange(x_rbf_final_dim[0]).reshape(15, 1)
    y_axis = x_rbf_final[:, 0]
    color = x_rbf_final[:, 1]

    plt.scatter(x_axis, y_axis, c=color)
    plt.show()


if __name__ == '__main__':
    main()
