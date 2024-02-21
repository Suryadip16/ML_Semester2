import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


def bag_reg(X, Y):

    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.3)
    max_depth = range(1, 8)
    score = 0
    for depth in max_depth:
        regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=depth, random_state=42), n_estimators=20)
        cv_scores = cross_val_score(regressor, x_train, y_train, cv=10)
        mean_score = cv_scores.mean()

        if mean_score > score:
            score = mean_score
            depth_val = depth
    regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=depth_val, random_state=42))
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    acc_score_bag = r2_score(y_test, y_pred)
    print(f"Accuracy with bagging: {acc_score_bag}")
    model = DecisionTreeRegressor(max_depth=depth_val)
    model.fit(x_train, y_train)
    # plot_tree(model)
    # plt.show()
    y_pred_dt = model.predict(x_test)
    acc_score_dt = r2_score(y_test, y_pred_dt)
    print(f"Accuracy without bagging: {acc_score_dt}")


def bag_classify(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.3)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    max_depth = range(1, 8)
    score = 0
    for depth in max_depth:
        clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=depth, random_state=42), n_estimators=20)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=10)
        mean_score = cv_scores.mean()

        if mean_score > score:
            score = mean_score
            depth_val = depth
    clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=depth_val, random_state=42))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc_score_bag = accuracy_score(y_test, y_pred)
    print(f"Accuracy with bagging: {acc_score_bag}")
    model = DecisionTreeClassifier(max_depth=depth_val)
    model.fit(x_train, y_train)
    # plot_tree(model)
    # plt.show()
    y_pred_dt = model.predict(x_test)
    acc_score_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Accuracy without bagging: {acc_score_dt}")


def main():
    reg_data_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    reg_X = reg_data_df.drop(["disease_score", "disease_score_fluct"], axis=1)
    reg_Y = reg_data_df["disease_score"]
    bag_reg(reg_X, reg_Y)
    log_data_df = pd.read_csv("wisconsin_breast_cancer.csv")
    log_Y = log_data_df['diagnosis']
    log_X = log_data_df.drop(["diagnosis"], axis=1)
    bag_classify(log_X, log_Y)


if __name__ == '__main__':
    main()


