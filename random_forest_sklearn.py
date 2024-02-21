import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree


def rf_classifier(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.3)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    # plt.figure(figsize=(15, 20))
    # plot_tree(clf)
    # plt.show()
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"Accuracy score for RF Classifier: {acc_score}")
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred_dt = model.predict(x_test)
    acc_score_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Accuracy of Decision Tree Classifier: {acc_score_dt}")


def rf_regressor(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.3)
    clf = RandomForestRegressor()
    clf.fit(x_train, y_train)
    # plt.figure(figsize=(15, 20))
    # plot_tree(clf)
    # plt.show()
    y_pred = clf.predict(x_test)
    acc_score = r2_score(y_test, y_pred)
    print(f"Accuracy score for RF Regressor: {acc_score}")
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_pred_dt = model.predict(x_test)
    acc_score_dt = r2_score(y_test, y_pred_dt)
    print(f"Accuracy of Decision Tree Regressor: {acc_score_dt}")


def main():
    rf_clf = pd.read_csv("wisconsin_breast_cancer.csv")
    rf_clf = rf_clf.dropna(axis=1, how='all')
    log_Y = rf_clf['diagnosis']
    log_X = rf_clf.drop(["diagnosis"], axis=1)
    rf_classifier(log_X, log_Y)

    rf_reg = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    rf_X = rf_reg.drop(["disease_score", "disease_score_fluct"], axis=1)
    rf_Y = rf_reg["disease_score"]
    rf_regressor(rf_X, rf_Y)


if __name__ == '__main__':
    main()

