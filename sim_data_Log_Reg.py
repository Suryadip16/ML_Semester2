import pandas as pd
import sklearn.model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def load_data():
    sim_df = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
    x = sim_df.drop(["disease_status"], axis=1)
    y = sim_df["disease_status"]
    return x, y
    # print(X)
    # print(y)


def split_n_pred(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, train_size=0.7, shuffle=True)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    print(acc_score)


def main():
    X, y = load_data()
    split_n_pred(X, y)


if __name__ == "__main__":
    main()
