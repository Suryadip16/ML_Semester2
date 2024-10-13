import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.compose import ColumnTransformer
import shap
import matplotlib.pyplot as plt

bike_df = pd.read_csv("Bikeshare.csv")
print(bike_df.columns)

# Good Way to Build a Heatmap (for continuous variables only):

numerical_cols = bike_df.select_dtypes(include=['number']).columns
corr_matrix = pd.DataFrame(bike_df[numerical_cols], columns=numerical_cols).corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt='.1g')
plt.show()


# bike_x = bike_df.drop((["Unnamed: 0", "registered", "casual", "bikers"]), axis=1)
# bike_y = bike_df['bikers']
# # print(bike_y)
# # print(bike_x)
#
# reg = xgb.XGBRegressor()
#
# x = bike_x.to_numpy()
# y = bike_y.to_numpy()

# One way to Do Cross Val in xgb:

# kf = KFold(n_splits=10, random_state=42, shuffle=True)
#
#
# cross_val_scores = []
# for fold, [train_index, test_index] in enumerate(kf.split(x)):
#     x_train, x_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     ct = ColumnTransformer(transformers=[("OHE", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 7])], remainder="passthrough")
#     ct.fit(x_train)
#     x_train = ct.transform(x_train)
#     x_test = ct.transform(x_test)
#     reg.fit(x_train, y_train)
#     y_pred = reg.predict(x_test)
#     score = r2_score(y_test, y_pred)
#     cross_val_scores.append(score)
# mean_score = np.mean(cross_val_scores)
# sd = np.std(cross_val_scores)
#
# print(mean_score)
# print(sd)

# Without np array

# Load the dataset
bike_df = pd.read_csv("Bikeshare.csv")

# Drop unnecessary columns
bike_x = bike_df.drop(["Unnamed: 0", "registered", "casual", "bikers"], axis=1)
bike_y = bike_df['bikers']

# Initialize XGBoost Regressor
reg = xgb.XGBRegressor()

# One way to Do Cross Validation with ColumnTransformer and XGBoost
kf = KFold(n_splits=10, random_state=42, shuffle=True)

cross_val_scores = []
for fold, [train_index, test_index] in enumerate(kf.split(bike_x, bike_y)):
    reg = xgb.XGBRegressor()
    x_train, x_test = bike_x.iloc[train_index], bike_x.iloc[test_index]
    y_train, y_test = bike_y.iloc[train_index], bike_y.iloc[test_index]

    #Define ColumnTransformer with OneHotEncoder for categorical columns
    ct = ColumnTransformer(transformers=[("OHE", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 7])], remainder="passthrough")

    # Fit and transform the training data
    x_train_transformed = ct.fit_transform(x_train)

    # Transform the test data
    x_test_transformed = ct.transform(x_test)

    # Fit XGBoost model on the transformed training data
    reg.fit(x_train_transformed, y_train)

    # Predict using the model
    y_pred = reg.predict(x_test_transformed)

    #Calculate R-squared score
    score = r2_score(y_test, y_pred)

    # Append score to list of cross-validation scores
    cross_val_scores.append(score)

# Calculate mean and standard deviation of cross-validation scores
mean_score = np.mean(cross_val_scores)
sd = np.std(cross_val_scores)

print(mean_score)
print(sd)

explainer = shap.Explainer(reg)
shap_values = explainer(x_test_transformed)
shap.plots.beeswarm(shap_values)

# shap_values = shap.Explainer(reg).shap_values(x_test_transformed)
# shap.summary_plot(shap_values, x_test_transformed)

# shap_values = shap.TreeExplainer(reg).shap_values(x_test_transformed)
# shap.summary_plot(shap_values, x_test_transformed)

#OR

# explainer = shap.Explainer(reg)
# shap_values = explainer(x_test_transformed)
# print(shap_values)
# shap.plots.beeswarm(shap_values[0])


# Another way

# regressor = xgb.XGBRegressor()
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, shuffle=True)
# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, shuffle=True)
# ct = ColumnTransformer(transformers=[("OHE", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 7])], remainder="passthrough")
# ct.fit(x_train)
# x_train = ct.transform(x_train)
# x_test = ct.transform(x_test)
# x_val = ct.transform(x_val)
#
# regressor.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=1000, eval_metric="logloss", early_stopping_rounds=30)
# y_pred = regressor.predict(x_test)
# score = r2_score(y_test, y_pred)
# print(score)
