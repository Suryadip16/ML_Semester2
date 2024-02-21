import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import pandas as pd

data_df = pd.read_csv("Life Expectancy Data.csv")

print(data_df.info())
si_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
si_min = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=min(data_df["GDP"]))

ct1 = ColumnTransformer(transformers=[('mean', si_mean, ["Life expectancy ", "Adult Mortality", "Alcohol", "Hepatitis B",
                                                         " BMI ", "Polio", "Total expenditure", "Diphtheria ", "Population",
                                                         " thinness  1-19 years", " thinness 5-9 years",
                                                         "Income composition of resources", "Schooling"]),
                                      ('min', si_min, ["GDP"])], remainder='passthrough')

data_np = ct1.fit_transform(data_df)
print(data_np)

life_y = data_np[:, 0]
life_X = data_np[:, 1:]

ct_2 = ColumnTransformer(transformers=[('ohe', OneHotEncoder(sparse_output=False), [13, 15]),
                                       ('ordinal', OrdinalEncoder(), [14])], remainder="passthrough")

life_x = ct_2.fit_transform(life_X)

x_train, x_test, y_train, y_test = train_test_split(life_x, life_y, test_size=0.3, shuffle=True)




# max_depth = range(1, 200)
# score = 0
# for depth in max_depth:
#     model = DecisionTreeRegressor(random_state=36, max_depth=depth)
#     cv_scores = cross_val_score(model, x_train, y_train)
#     cv_score_mean = cv_scores.mean()
#     if cv_score_mean > score:
#         score = cv_score_mean
#         use_depth = depth

model = DecisionTreeRegressor(random_state=36, max_depth=10)

#plt.figure(figsize=(20, 25))
model.fit(x_train, y_train)
#plot_tree(model)
#plt.show()

y_pred = model.predict(x_test)
acc_score = r2_score(y_test, y_pred)
print(acc_score)












