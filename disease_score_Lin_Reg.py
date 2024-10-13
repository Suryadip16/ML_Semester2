import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def load_data():
    disease_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    x = disease_df.drop(["disease_score", "disease_score_fluct"], axis=1)
    y = disease_df["disease_score"]
    return x, y


def split_train(X, Y):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, train_size=0.7, shuffle=True)
    # print(X_train)
    # print(X_test)
    # print(Y_train)
    # print(Y_test)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    print(f"The coefficients are {model.coef_}")
    print(f"The score is {model.score(X_train, Y_train)}")
    pred = model.predict(X_test)
    score = r2_score(Y_test, pred)
    print(f"The r2_score for the model is {score}")
    if score >= 0.75:
        print("This model is working satisfactorily.")
    else:
        print("The model needs to be changed")


def main():
    X, Y = load_data()
    split_train(X, Y)


if __name__=="__main__":
    main()






