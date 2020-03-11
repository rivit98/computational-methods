from mylib import *
import numpy as np


def f(x):
    return float64(1/(1+x**2))


def poly(x, a):
    y = 0
    for index, val in enumerate(a):
        y += val * (x**index)

    return y


def inter(n):
    orig_x = np.linspace(-5, 5, 100)
    plt.plot(orig_x, [f(x) for x in orig_x])

    interval = np.linspace(-5, 5, n + 1)
    f_values = [f(x) for x in interval]
    vander = np.vander(interval, increasing=True)
    a = np.linalg.solve(vander, f_values)

    # print(vander)
    # print(a)

    plt.plot(orig_x, [poly(x, a) for x in orig_x])

    # b) error
    plt.plot(orig_x, [abs(f(x) - poly(x, a)) for x in orig_x])
    plt.show()


def task1():
    n_ = [15]

    for n in n_:
        inter(n)


if __name__ == "__main__":
    task1()
