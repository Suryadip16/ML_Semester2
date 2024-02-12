from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


def load_data():
    #Load Data
    sonar_df = pd.read_csv("Sonar.csv", header=None)

    #Convert to np array
    sonar_np = sonar_df.to_numpy()
    print(sonar_np.shape[1])

    #divide into x and y
    sonar_x = sonar_np[:, :sonar_np.shape[1] - 2]
    sonar_y = sonar_np[:, sonar_np.shape[1] - 1]
    print(sonar_y.shape)
    print(sonar_x.shape)
    return sonar_x, sonar_y


def data_encoding(np_array):
    #Label encoding to convert R and M vals to 0 and 1
    sonar_y = LabelEncoder().fit_transform(np_array)
    #print(sonar_y)
    return sonar_y


def Train_test_split(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    return x_train, x_test, y_train, y_test


def data_normalization(x_train, x_test):
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_normalized = scaler.transform(x_train)
    #y_train_normalized = scaler.transform(y_train)
    x_test_normalized = scaler.transform(x_test)
    #y_test_normalized = scaler.transform(y_test)
    return x_train_normalized, x_test_normalized


def data_standardization(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_standardized = scaler.transform(x_train)
    #y_train_standardized = scaler.transform(y_train)
    x_test_standardized = scaler.transform(x_test)
    #y_test_standardized = scaler.transform(y_test)
    return x_train_standardized, x_test_standardized


def log_reg(x_train_preprocessed, x_test_preprocessed, y_train, y_test):
    model = LogisticRegression()
    model.fit(x_train_preprocessed, y_train)
    y_pred = model.predict(x_test_preprocessed)
    score = accuracy_score(y_test, y_pred)
    return score


def main():
    X, Y = load_data()
    y = data_encoding(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    x_train_normalized, x_test_normalized = data_normalization(x_train, x_test)
    x_train_standardized, x_test_standardized = data_standardization(x_train, x_test)
    score1 = log_reg(x_train_normalized, x_test_normalized, y_train, y_test)
    score2 = log_reg(x_train_standardized, x_test_standardized, y_train, y_test)
    print(f"Accuracy with Data Normalization: {score1}")
    print(f"Accuracy with Data Standardization: {score2}")


if __name__ == '__main__':
    main()

















