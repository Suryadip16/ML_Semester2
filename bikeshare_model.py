import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split


def EDA():
    bike_df = pd.read_csv("Bikeshare.csv")
    print(bike_df.info())
    # print(bike_df["registered"].corr(bike_df["bikers"]))
    # numerical_cols = bike_df.select_dtypes(include=['number']).columns
    # cocf_list = []
    # for col in numerical_cols:
    #     cocf = bike_df[col].corr(bike_df["bikers"])
    #     cocf_list.append(cocf)
    # plt.bar(numerical_cols, cocf_list)
    # plt.show()

    # print(bike_df["holiday"].unique())
    # plt.bar(bike_df["holiday"], bike_df['bikers'])
    # plt.show()
    #
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
    # plt.bar(bike_df["registered"], bike_df['bikers'])
    # plt.title("registered")
    # plt.show()
    # missing_values = bike_df.isnull().sum()
    # print(missing_values)  # No missing values found
    #
    # # Important EDA Findings:
    # plt.bar(bike_df["season"], bike_df['bikers'])
    # plt.show()  # Season 1 is considerably lower tha rest of the seasons.
    # print(bike_df["temp"].corr(bike_df["atemp"]))  # 0.992 corr b/w them. One can be removed.
    #
    # plt.bar(bike_df["hr"], bike_df['bikers'])
    # plt.show()  # Less bikers during the Dawn hours of 3-5. Maximum during peak office hours and afternoon.
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
        # scaler = StandardScaler()
        # scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

        month_categories = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
        weather_categories = ["cloudy/misty", "light rain/snow", "clear", "heavy rain/snow"]
        ct = ColumnTransformer(transformers=[('OHE1', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]),
                                             ('OHE2', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [7]),
                                             ],
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
    regressor = RandomForestRegressor(n_estimators=200)
    cv_scores = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ct = ColumnTransformer(
            transformers=[('OHE', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 7]),
                          ('ss', StandardScaler(), [2, 3, 5, 8, 9, 10])],
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
        x_train2, x_val, y_train2, y_val = train_test_split(x_train1, y_train1, shuffle=True, random_state=40,
                                                            test_size=0.3)
        ct = ColumnTransformer(transformers=[('OHE1', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]),
                                             (
                                                 'OHE2', OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                                                 [7]), ('ss', StandardScaler(), [2, 3, 5, 8, 9, 10])],
                               remainder="passthrough")

        ct.fit(x_train2)
        x_train1 = ct.transform(x_train1)
        x_test = ct.transform(x_test)
        x_train2 = ct.transform(x_train2)
        x_val = ct.transform(x_val)

        depth_score = 0
        for depth in range(1, 20):
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
    regressor = xgb.XGBRegressor(objective="reg:linear")
    cv_scores = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ct = ColumnTransformer(transformers=[('OHE1', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]),
                                             (
                                                 'OHE2', OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                                                 [7])],
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


def gradient_boost(x, y):
    kf = KFold(n_splits=10, random_state=40, shuffle=True)
    regressor = GradientBoostingRegressor()
    cv_scores = []
    for fold, [train_index, test_index] in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ct = ColumnTransformer(transformers=[('OHE1', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1]),
                                             (
                                                 'OHE2', OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                                                 [7])],
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
    #score, sd = lin_reg(bike_x, bike_y)
    #score_rf, sd_rf = random_forest(bike_x, bike_y)
    score_dt, sd_dt = Decision_tree(bike_x, bike_y)
    score_xgb, sd_xgb = xg_boost(bike_x, bike_y)
    score_gb, sd_gb = gradient_boost(bike_x, bike_y)
    # print("Linear Regression:")
    # print(score)
    # print(sd)
    # print("Random Forest:")
    # print(score_rf)
    # print(sd_rf)
    print("Decision Tree:")
    print(score_dt)
    print(sd_dt)
    print("XGBoost:")
    print(score_xgb)
    print(sd_xgb)
    print("GBoost:")
    print(score_gb)
    print(sd_gb)
    x_train, x_test, y_train, y_test = train_test_split(bike_x, bike_y)
    gb_regressor = GradientBoostingRegressor(random_state=42)
    gb_regressor.fit(x_train, y_train)
    gb_y_pred = gb_regressor.predict(x_test)
    gb_score = r2_score(y_test, gb_y_pred)
    print(gb_score)


if __name__ == '__main__':
    main()

# model_params = {
#         'random_forest': {
#         'model': RandomForestRegressor(),
#         'params': {
#             'n_estimators': [1, 5, 10]
#         }
#     },
#
#             "XgBoost" : {
#                 'model' : xgb.XGBRegressor(),
#                 'params' : {
#                     'learning_rate' : [0.01, 0.1, 0.2, 0.3],
#                     'n_estimators': [10, 20, 20,50, 70],
#                     'gamma': [0, 0.1, 0.2, 0.3],
#                     'max_depth': [3, 5, 7, 10]
#             }
#     },
#
#
#
# }
#
#
# scores = []
#
# for model_name, mp in model_params.items():
#     clf = GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
#     clf.fit(X_train, y_train)
#     scores.append({
#         'model': model_name,
#         'best_score': clf.best_score_,
#         'best_params': clf.best_params_
#     })
#
# df2 = pd.DataFrame(scores, columns=['model', 'best_score','best_params'])
# print(df2)
# Dealing with missing values xgboost style:
# Identifying Missing values:
# print(df.isnull().sum())


#GradientBoostingClassifier(*, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3)

