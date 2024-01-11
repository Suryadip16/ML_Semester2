import numpy as np
import matplotlib.pyplot as plt
import math


def implement_square_matrix(matrix):
    matrix_transpose = matrix.T
    square_matrix = matrix_transpose @ matrix
    print(square_matrix)


def implement_linear_func(a0, a1, x):
    y = []
    for i in range(len(x)):
        d = a1*x[i] + a0
        y.append(d)
    print(y)
    print(x)
    plt.plot(x, y)
    plt.title("Linear Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def implement_quadratic_function(a0, a1, a2, x):
    y = []
    for i in range(len(x)):
        d = a0 + a1*x[i] + a2*x[i]**2
        y.append(d)
    print(x)
    print(y)
    plt.plot(x, y)
    plt.title("Quadratic Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def gaussian_pdf(mean, sd, x):
    gpdf = []
    for i in range(len(x)):
        result = (math.exp(-0.5*((x[i] - mean) / sd)**2)) / (sd * ((2 * math.pi) ** 0.5))
        gpdf.append(result)
    print(gpdf)
    print(x)
    plt.plot(x, gpdf)
    plt.title("Gaussian PDF")
    plt.xlabel("x")
    plt.ylabel("gpdf")
    plt.show()


def function_derivative(x):
    y_values = []
    for i in range(len(x)):
        y = x[i] ** 2
        y_values.append(y)
    dx = x[1] - x[0]
    f_prime = np.gradient(y_values, dx)
    plt.plot(x, y_values, color="Red", label="Function f(x)")
    plt.plot(x, f_prime, color="Blue", label="f'(x)")
    plt.title("Function and Its Derivative vs x values")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def main():
    print("Square Matrix:")
    #Q1
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    implement_square_matrix(matrix)

    print("Linear Function:")
    #Q2
    a0 = 3
    a1 = 2
    x = np.linspace(-100, 100, 100)
    implement_linear_func(a0, a1, x)

    print("Quadratic Function:")
    #Q3
    a0 = 4
    a1 = 3
    a2 = 2
    x = np.linspace(-10, 10, 100)
    implement_quadratic_function(a0, a1, a2, x)


    print("Gaussian PDF:")
    #Q4
    mean = 0
    sigma = 15
    x = np.linspace(-100, 100, 100)
    gaussian_pdf(mean, sigma, x)


    print("Function Derivative:")
    #Q5
    x = np.linspace(-100, 100, 100)
    function_derivative(x)


if __name__ == "__main__":
    main()



