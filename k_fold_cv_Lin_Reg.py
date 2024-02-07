import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
sim_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
x = sim_df.drop(["disease_score","disease_score_fluct"], axis=1)
y = sim_df["disease_score"]
equal_parts_x = np.array_split(x.index, 10)
# print(equal_parts[3])
# print(len(equal_parts))
acc = []

for i in equal_parts_x:
    fold_x_test = x.iloc[i]
    fold_y_test = y.iloc[i]
    fold_x_train = x.drop(i)
    fold_y_train = y.drop(i)
    model = LinearRegression()
    model.fit(fold_x_train, fold_y_train)
    pred = model.predict(fold_x_test)
    a = r2_score(fold_y_test, pred)
    acc.append(a)

print(acc)
accuracy = sum(acc)/len(acc)
print(accuracy)

    # print(fold_y_train)
    # print(fold_x_train)
    # # print(fold_x_test)
    # # print(fold_y_test)














