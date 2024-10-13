import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

from sklearn.model_selection import train_test_split


def EDA():
    bike_df = pd.read_csv("Bikeshare.csv")
    print(bike_df.info())
    print(bike_df["registered"].corr(bike_df["bikers"]))
    numerical_cols = bike_df.select_dtypes(include=['number']).columns
    cocf_list = []
    # for col in numerical_cols:
    #     cocf = bike_df[col].corr(bike_df["bikers"])
    #     cocf_list.append(cocf)
    # plt.bar(numerical_cols, cocf_list)
    # plt.show()

    # print(bike_df["holiday"].unique())
    # plt.bar(bike_df["holiday"], bike_df['bikers'])
    # plt.show()

    # print(bike_df["holiday"].unique())
    # plt.bar(bike_df["holiday"], bike_df['bikers'])
    # plt.title("Holiday")
    # plt.show()
    #
    # print(bike_df["weekday"].unique())
    # plt.bar(bike_df["weekday"], bike_df['bikers'])
    # plt.title("Weekday")
    # plt.show()
    #
    # print(bike_df["workingday"].unique())
    # plt.bar(bike_df["workingday"], bike_df['bikers'])
    # plt.title("workingday")
    # plt.show()
    #
    # print(bike_df["weathersit"].unique())
    # plt.bar(bike_df["weathersit"], bike_df['bikers'])
    # plt.title("weathersit")
    # plt.show()
    #
    # print(bike_df["temp"].unique())
    # plt.bar(bike_df["temp"], bike_df['bikers'])
    # plt.title("temp")
    # plt.show()
    #
    # print(bike_df["atemp"].unique())
    # plt.bar(bike_df["atemp"], bike_df['bikers'])
    # plt.title("atemp")
    # plt.show()
    #
    # print(bike_df["casual"].unique())
    # plt.bar(bike_df["casual"], bike_df['bikers'])
    # plt.title("casual")
    # plt.show()
    # #
    # print(bike_df["registered"].unique())
    # plt.plot(bike_df["registered"], bike_df['bikers'])
    # plt.title("registered")
    # plt.show()
    missing_values = bike_df.isnull().sum()
    print(missing_values)  # No missing values found

    # Important EDA Findings:
    # plt.bar(bike_df["season"], bike_df['bikers'])
    # plt.show()  # Season 1 is considerably lower tha rest of the seasons.
    # print(bike_df["temp"].corr(bike_df["atemp"])) # 0.992 corr b/w them. One can be removed.
    #
    # plt.bar(bike_df["hr"], bike_df['bikers'])
    # plt.show()  #Less bikers during the Dawn hours of 3-5. Maximum during peak office hours and afternoon.
    bike_X = bike_df.drop((["bikers", "Unnamed: 0", "casual", "registered", "atemp"]),
                          axis=1)  # atemp and temp have very high corr. casual + registered is basically equal to the target variable.
    bike_Y = bike_df["bikers"]
    bike_y = bike_Y.to_numpy()
    bike_x = bike_X.to_numpy()
    print(bike_x)
    return bike_x, bike_y


def lin_reg(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    regressor = LinearRegression()
    cv_scores = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        month_categories = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
        weather_categories = ["cloudy/misty", "light rain/snow", "clear", "heavy rain/snow"]
        ct = ColumnTransformer(transformers=[('OHE1', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]), ('OHE2', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [7])],
                               remainder="passthrough")
        ct.fit(x_train)
        x_train = ct.transform(x_train)
        x_test = ct.transform(x_test)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    mean_score = sum(cv_scores) / len(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


def random_forest(x, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=40)
    regressor = RandomForestRegressor(n_estimators=100)
    cv_scores = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ct = ColumnTransformer(transformers=[('OHE', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 7])],
                               remainder="passthrough")
        ct.fit(x_train)
        x_train = ct.transform(x_train)
        x_test = ct.transform(x_test)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    mean_score = sum(cv_scores) / len(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd


def Decision_tree(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    fold_acc_scores_list = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
            x_train1, x_test = x[train_index], x[test_index]
            y_train1, y_test = y[train_index], y[test_index]
            x_train2, x_val, y_train2, y_val = train_test_split(x_train1, y_train1, shuffle=True, random_state=40, test_size=0.7)
            ct = ColumnTransformer(transformers=[('OHE1', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]),
                                                 ('OHE2', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [7])],
                                   remainder="passthrough")
            ct.fit(x_train1)
            x_train1 = ct.transform(x_train1)
            x_test = ct.transform(x_test)
            x_train2 = ct.transform(x_train2)
            x_val = ct.transform(x_val)


            depth_score = 0
            for depth in range(1, 15):
                regressor = DecisionTreeRegressor(max_depth=depth, random_state=40)
                regressor.fit(x_train2, y_train2)
                val_pred = regressor.predict(x_val)
                val_acc = r2_score(y_val, val_pred)
                if val_acc > depth_score:
                    depth_score = val_acc
                    depth_value = depth
            regressor = DecisionTreeRegressor(max_depth=depth_value, random_state=40)
            regressor.fit(x_train1, y_train1)
            test_pred = regressor.predict(x_test)
            fold_acc_score = r2_score(y_test, test_pred)
            fold_acc_scores_list.append(fold_acc_score)
    mean_acc_score = np.mean(fold_acc_scores_list)
    sd_acc_score = np.std(fold_acc_scores_list)
    return mean_acc_score, sd_acc_score


def xg_boost(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    regressor = xgb.XGBRegressor()
    cv_scores = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ct = ColumnTransformer(transformers=[('OHE1', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]),
                                             (
                                             'OHE2', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [7])],
                               remainder="passthrough")
        ct.fit(x_train)
        x_train = ct.transform(x_train)
        x_test = ct.transform(x_test)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    mean_score = sum(cv_scores) / len(cv_scores)
    sd = np.std(cv_scores)
    return mean_score, sd




def main():
    bike_x, bike_y = EDA()

    # bike_x_use = normalization(bike_x)
    score, sd = lin_reg(bike_x, bike_y)
    score_rf, sd_rf = random_forest(bike_x, bike_y)
    score_dt, sd_dt = Decision_tree(bike_x, bike_y)
    score_xgb, sd_xgb = xg_boost(bike_x, bike_y)
    print("Linear Regression:")
    print(score)
    print(sd)
    print("Random Forest:")
    print(score_rf)
    print(sd_rf)
    print("Decision Tree:")
    print(score_dt)
    print(sd_dt)
    print("XGB:")
    print(score_xgb)
    print(sd_xgb)


if __name__ == '__main__':
    main()
