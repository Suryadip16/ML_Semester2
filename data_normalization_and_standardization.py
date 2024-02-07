import numpy as np
import pandas as pd


def load_data():
    disease_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    x = disease_df.drop(["disease_score", "disease_score_fluct"], axis=1)
    x_np = x.to_numpy()
    x_main = x_np.T
    return x_main


def normalization(np_array):
    list_main = []
    for row in np_array:
        min = np.min(row)
        max = np.max(row)
        list = []
        for val in row:
            new_val = (val - min)/(max - min)
            list.append(new_val)
        list_main.append(list)
    print(list_main)
    normalized_data = np.array(list_main)
    normalized_data_main = normalized_data.T
    print(normalized_data_main)


def standardization(np_array):
    list_main = []
    for row in np_array:
        mu = np.mean(row)
        sigma = np.std(row)
        list = []
        for val in row:
            new_val = (val - mu)/sigma
            list.append(new_val)
        list_main.append(list)
    data = np.array(list_main)
    for row in data:
        print(np.mean(row))
    #print(data)
    standardized_data = data.T
    print(standardized_data)


def main():
    x = load_data()
    normalization(x)
    standardization(x)


if __name__ == '__main__':
    main()
