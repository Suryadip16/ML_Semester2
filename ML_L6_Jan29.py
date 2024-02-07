import numpy as np
import matplotlib.pyplot as plt


def sigmoid_fn(x):
    result = []
    for i in x:
        s = 1 / (1 + np.exp(-i))
        result.append(s)

    print(result)
    plt.plot(x, result)
    plt.xlabel("x")
    plt.ylabel("Sigmoid Value")
    plt.show()


def sig_der(x):
    result = []
    dx = x[1] - x[0]
    for i in x:
        s = 1 / (1 + np.exp(-i))
        result.append(s)

    s_prime = np.gradient(result, dx)
    plt.plot(x, s_prime)
    plt.plot(x, result)
    plt.show()


def main():

    # Sigmoid Function
    x = np.linspace(-10, 10, 100)
    sigmoid_fn(x)

    # Sigmoid Derivative:
    sig_der(x)


if __name__ == "__main__":
    main()


