import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def load_data():
    disease_fluct_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    x = disease_fluct_df.drop(["disease_score", "disease_score_fluct"], axis=1)
    y = disease_fluct_df["disease_score_fluct"]
    return x, y


def split_train(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7, shuffle=True)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    print(f"The coefficients are {model.coef_}")
    pred = model.predict(X_test)
    score = r2_score(Y_test, pred)
    print(f"The r2_score is {score}")
    if score >= 0.75:
        print("This model is working satisfactorily.")
    else:
        print("The model needs to be changed")


def main():
    X, Y = load_data()
    split_train(X, Y)


if __name__ == "__main__":
    main()

