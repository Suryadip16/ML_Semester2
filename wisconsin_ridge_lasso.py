import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data():
    # Load Data
    breast_cancer_df = pd.read_csv("wisconsin_breast_cancer.csv")
    print(breast_cancer_df.info())
    print(breast_cancer_df['Unnamed: 32'])
    nan_columns = breast_cancer_df.columns[breast_cancer_df.isna().any()].tolist()
    print(nan_columns)
    breast_cancer_x = breast_cancer_df.drop(columns=["diagnosis", "id", "Unnamed: 32"])
    breast_cancer_y = breast_cancer_df["diagnosis"]
    return breast_cancer_x, breast_cancer_y


def split_data(x, y):
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, shuffle=True)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True)
    return x_train, x_val, x_test, y_train, y_val, y_test


def data_encoding(y_train, y_val, y_test):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    y_val = encoder.transform(y_val)
    return y_train, y_val, y_test


def data_normalization(x_train, x_val, x_test):
    normalizer = MinMaxScaler()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)
    x_val = normalizer.transform(x_val)
    return x_train, x_val, x_test


def ridge(x_train, x_val, x_test, y_train, y_val, y_test):
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100]
    new_score = 0
    for a in alpha_values:
        model = Ridge(alpha=a)
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        y_val_pred2 = []
        for val in y_val_pred:
            if val < 0.5:
                y_val_pred2.append(0)
            else:
                y_val_pred2.append(1)

        score = accuracy_score(y_val, y_val_pred2)
        print(f"Accuracy of {score} with alpha = {a}")

        if score > new_score:
            new_score = score
            alpha_val = a

    model = Ridge(alpha=alpha_val)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_predict = []
    for val in y_pred:
        if val < 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    accuracy = accuracy_score(y_test, y_predict)
    print(f'Final Accuracy(Ridge): {accuracy} with alpha as {alpha_val}')


def lasso(x_train, x_val, x_test, y_train, y_val, y_test):
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100]
    new_score = 0
    for a in alpha_values:
        model = Lasso(alpha=a)
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        y_val_pred2 = []
        for val in y_val_pred:
            if val < 0.5:
                y_val_pred2.append(0)
            else:
                y_val_pred2.append(1)

        score = accuracy_score(y_val, y_val_pred2)
        print(f"Accuracy of {score} with alpha = {a}")

        if score > new_score:
            new_score = score
            alpha_val = a

    model = Ridge(alpha=alpha_val)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_predict = []
    for val in y_pred:
        if val < 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    accuracy = accuracy_score(y_test, y_predict)
    print(f'Final Accuracy(Lasso): {accuracy} with alpha as {alpha_val}')


def main():
    X, Y = load_data()
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y)
    y_train, y_val, y_test = data_encoding(Y_train, Y_val, Y_test)
    x_train, x_val, x_test = data_normalization(X_train, X_val, X_test)
    ridge(x_train, x_val, x_test, y_train, y_val, y_test)
    lasso(x_train, x_val, x_test, y_train, y_val, y_test)


if __name__ == '__main__':
    main()
