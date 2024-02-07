import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sim_df = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
x = sim_df.drop(["disease_status"], axis=1)
y = sim_df["disease_status"]
equal_parts_x = np.array_split(x.index, 10)
# print(equal_parts[3])
# print(len(equal_parts))
acc = []

for i in equal_parts_x:
    fold_x_test = x.iloc[i]
    fold_y_test = y.iloc[i]
    fold_x_train = x.drop(i)
    fold_y_train = y.drop(i)
    model = LogisticRegression()
    model.fit(fold_x_train, fold_y_train)
    pred = model.predict(fold_x_test)
    a = accuracy_score(fold_y_test, pred)
    acc.append(a)

print(acc)
accuracy = sum(acc)/len(acc)
print(accuracy)

    # print(fold_y_train)
    # print(fold_x_train)
    # # print(fold_x_test)
    # # print(fold_y_test)














