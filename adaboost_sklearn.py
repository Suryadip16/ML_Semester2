import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier #n_estimators=50, learning_rate=1.0
from sklearn.ensemble import AdaBoostRegressor #n_estimators=50, learning_rate=1.0, loss='linear' or "square" or "exponential"
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


def adaboost_reg(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.2, random_state=42)
    regressor = AdaBoostRegressor(n_estimators=200, random_state=42)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    acc_score = r2_score(y_test, y_pred)
    print(f"Accuracy for Adaboost Regressor: {acc_score}")
    max_depth = range(1, 8)
    score = 0
    for depth in max_depth:
        regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=depth, random_state=42), n_estimators=200, random_state=42, oob_score=True)
        equal_parts_of_X = np.array_split(X.index, 10)
        # print(equal_parts_of_X)
        cv_acc = []
        for i in equal_parts_of_X:
            fold_x_train = X.drop(i)
            fold_y_train = Y.drop(i)
            x_train2, x_val, y_train2, y_val = train_test_split(fold_x_train, fold_y_train, test_size=0.2, random_state=42)
            regressor.fit(x_train2, y_train2)
            pred = regressor.predict(x_val)
            a = r2_score(y_val, pred)
            cv_acc.append(a)
        mean_score = sum(cv_acc) / len(cv_acc)
        if mean_score > score:
            score = mean_score
            depth_val = depth

    regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=depth_val, random_state=42), n_estimators=200, random_state=42)
    regressor.fit(x_train, y_train)
    y_pred_bag = regressor.predict(x_test)
    acc_score_bag = r2_score(y_test, y_pred_bag)
    print(f"Accuracy with bagging: {acc_score_bag}")


def adaboost_clf(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.3, random_state=42)
    # encoder = LabelEncoder()
    # encoder.fit(y_train)
    # y_train = encoder.transform(y_train)
    # y_test = encoder.transform(y_test)
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"Accuracy for Adaboost Classifier: {acc_score}")
    max_depth = range(1, 8)
    score = 0
    for depth in max_depth:
        clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=depth, random_state=42), n_estimators=200, random_state=42)
        equal_parts_of_X = np.array_split(X.index, 10)
        # print(equal_parts_of_X)
        cv_acc = []
        for i in equal_parts_of_X:
            fold_x_train = X.drop(i)
            fold_y_train = Y.drop(i)
            x_train2, x_val, y_train2, y_val = train_test_split(fold_x_train, fold_y_train, test_size=0.2, random_state=42)
            clf.fit(x_train2, y_train2)
            pred = clf.predict(x_val)
            a = accuracy_score(y_val, pred)
            cv_acc.append(a)
        mean_score = sum(cv_acc) / len(cv_acc)
        if mean_score > score:
            score = mean_score
            depth_val = depth
    clf_bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=depth_val, random_state=42), n_estimators=200, random_state=42)
    clf_bag.fit(x_train, y_train)
    y_pred_bag = clf_bag.predict(x_test)
    acc_score_bag = accuracy_score(y_test, y_pred_bag)
    print(f"Accuracy with bagging: {acc_score_bag}")


def main():
    print("Wisconsin breast Cancer")
    ada_clf = pd.read_csv("wisconsin_breast_cancer.csv")
    ada_clf = ada_clf.dropna(axis=1, how='all')
    ada_Y = ada_clf['diagnosis']
    ada_X = ada_clf.drop(["diagnosis"], axis=1)
    adaboost_clf(ada_X, ada_Y)

    print("simulated_data_multiple_linear_regression_for_ML")
    ada_reg = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    ada_reg_X = ada_reg.drop(["disease_score", "disease_score_fluct"], axis=1)
    ada_reg_Y = ada_reg["disease_score"]
    adaboost_reg(ada_reg_X, ada_reg_Y)




if __name__ == '__main__':
    main()
