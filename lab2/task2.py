from mylib import *
import numpy as np
from math import cos, pi


def f(x):
    return float64(1/(1+x**2))


def poly(x, a):
    y = 0
    for index, val in enumerate(a):
        y += val * (x**index)

    return y


def x_k(a, b, k, n):
    return (1/2 * (a + b)) + (1/2 * (b - a)) * cos((((2 * k) - 1) / (2 * n)) * pi)


def inter(n):
    a = -5
    b = 5
    orig_x = np.linspace(a, b, 100)
    plt.plot(orig_x, [f(x) for x in orig_x])

    interval = [x_k(a, b, k, n) for k in range(1, n+1)]
    f_values = [f(x) for x in interval]
    vander = np.vander(interval, increasing=True)
    a = np.linalg.solve(vander, f_values)

    # print(vander)
    # print(a)

    plt.plot(orig_x, [poly(x, a) for x in orig_x])

    # b) error
    plt.plot(orig_x, [abs(f(x) - poly(x, a)) for x in orig_x])
    plt.show()


def task2():
    n_ = [15]

    for n in n_:
        inter(n)


if __name__ == "__main__":
    task2()
