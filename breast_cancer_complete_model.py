import pandas
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

#Load Data:

breast_cancer_df = pandas.read_csv("breast-cancer.csv")
breast_cancer_np = breast_cancer_df.to_numpy()

ct = ColumnTransformer(transformers=[('OHE', OneHotEncoder(sparse_output=False), [1, 4, 6, 7, 8]),
                                     ('OE', OrdinalEncoder(), [0, 2, 3, 5])], remainder='passthrough')
breast_cancer_np_encoded = ct.fit_transform(breast_cancer_np)
print(breast_cancer_np_encoded.shape)
idx = breast_cancer_np_encoded.shape[1] - 1


breast_cancer_x = breast_cancer_np_encoded[:, :idx]
breast_cancer_Y = breast_cancer_np_encoded[:, idx]
label_encoder = LabelEncoder()
breast_cancer_y = label_encoder.fit_transform(breast_cancer_Y)

x_train, x_temp, y_train, y_temp = train_test_split(breast_cancer_x, breast_cancer_y, shuffle=True, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, shuffle=True, test_size=0.5)

#Logistic Regresion
model = LogisticRegression(penalty='l2')
model.fit(x_train, y_train)
y_p = model.predict(x_test)
acc = accuracy_score(y_test, y_p)
print(acc)




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




