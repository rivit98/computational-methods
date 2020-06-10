from nodepy.runge_kutta_method import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mtplt


def rk4(x, h, a, b, f, theory_f=None):
    # n = math.floor((b - a) / h)
    ys = []
    xs = np.arange(a, b + h, h)

    for t in xs:
        ys.append(x)
        k1 = h * f(t, x)
        k2 = h * f(t + (h / 2), x + (k1 / 2))
        k3 = h * f(t + (h / 2), x + (k2 / 2))
        k4 = h * f(t + h, x + k3)
        x += (k1 + (2 * k2) + (2 * k3) + k4) / 6

    if theory_f:
        plt.subplot(1, 2, 1)
    plt.plot(xs, ys)  # solved
    plt.grid()
    # plt.legend(["Solved"])
    plt.title("Solved function")

    if theory_f:
        yy = [theory_f(x) for x in xs]
        plt.subplot(1, 2, 2)
        plt.plot(xs, yy, c="red")  # expected
        # plt.legend(["Expected"])
        plt.title("Expected function")
        plt.grid()

    plt.show()
    print("[solved]   x({}) = {}".format(b, ys[-1]))

    if theory_f:
        print("[expected] x({}) = {}".format(b, yy[-1]))
        print("error: {}".format(np.fabs(ys[-1] - yy[-1])))
        plt.plot(xs, [np.fabs(i - j) for i, j in zip(ys, yy)])
        # plt.legend(["Error"])
        plt.title("Error for step={}".format(h))
        plt.grid()
        plt.show()


def subtask_1():
    def f(t, x):
        if np.isclose(t, 0.0) and np.isclose(x, 0.0):
            return 0

        return (x / t) + (t * (1 / np.cos(x / t)))

    a = 0
    b = 1
    h = 2 ** (-7)
    x = 0

    rk4(x, h, a, b, f, lambda x: x * np.arcsin(x))


def subtask_2():
    a = 0
    b = 3
    hh = [0.015, 0.02, 0.025, 0.03]
    x = 0

    def f(t, x):
        return 100 * (np.sin(t) - x)

    def expected_f(x):
        return ((100 * np.power(np.e, -100 * x)) +
                (10000 * np.sin(x)) -
                (100 * np.cos(x))) / 10001

    for h in hh:
        rk4(x, h, a, b, f, expected_f)

    def stability_func(h, l):
        z = h * l
        return 1 + z + (1 / 2 * np.power(z, 2)) + (1 / 6 * np.power(z, 3)) + (1 / 24 * np.power(z, 4))

    def test_stability(h, l):
        print("h={} 𝜆={}: stable? {}".format(h, l, "Yes" if np.fabs(stability_func(h, l)) < 1 else "No"))

    for h in hh:
        test_stability(h, -100)

    rk44 = loadRKM('RK44')
    rk44.plot_stability_region(bounds=[-4, 1, -3, 3], alpha=0.5, color="blue")
    plt.xlabel(r"$Re(\xi)$")
    plt.ylabel(r"$Im(\xi)$")
    plt.show()


def task1():
    subtask_1()
    subtask_2()


def rk45(f, t, x, h):
    c20, c21 = 0.25, 0.25
    c30, c31, c32 = 0.375, 0.09375, 0.28125
    c40, c41, c42, c43 = 12/13, 1932/2197, -7200/2197, 7296/2197
    c51, c52, c53, c54 = 439/216, -8, 3680/513, -845/4104
    c60, c61, c62, c63, c64, c65 = 0.5, -8/27, 2, -3544/2565, 1859/4104, -0.275
    a1, a2, a3, a4, a5 = 25/216, 0, 1408/2565, 2197/4104, -0.2
    b1, b2, b3, b4, b5, b6 = 16/135, 0, 6656/12825, 28561/56430, -0.18, 2/55

    k1 = h * f(t, x)
    k2 = h * f(t + c20 * h, x + c21 * k1)
    k3 = h * f(t + c30 * h, x + c31 * k1 + c32 * k2)
    k4 = h * f(t + c40 * h, x + c41 * k1 + c42 * k2 + c43 * k3)
    k5 = h * f(t + h, x + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4)
    k6 = h * f(t + c60 * h, x + c61 * k1 + c62 * k2 + c63 * k3 + c64 * k4 + c65 * k5)
    x5 = x + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
    x = x + a1 * k1 + a3 * k3 + a4 * k4 + a5 * k5
    e = abs(x - x5)
    return x, e


def rk45ad(f, t, x, h, tb, emin, emax, theory_f, hmin=1.0e-3, hmax=1.0e-1, itmax=1000):
    # print("n     h     t     x")
    # print("0   {:.5f}   {:.5f}   {:.5f}   0".format(h, t, x))

    delta = 0.5e-5
    k = 0
    iflag = 1
    ts, xs = [t], [x]

    def sign(v, s):
        if s < 0:
            return -v
        return v

    while k < itmax:
        k += 1
        if np.fabs(h) < hmin:
            h = sign(1, h) * hmin

        if np.fabs(h) > hmax:
            h = sign(1, h) * hmax

        d = np.fabs(tb - t)
        if d <= np.fabs(h):
            iflag = 0
            if d <= delta * max(np.fabs(tb), np.fabs(t)):
                break
            h = sign(1.0, h) * d

        xsave = x
        tsave = t
        x, e = rk45(f, t, x, h)
        t += h
        # print("{}   {:.5f}   {:.5f}   {:.5f}   {}".format(k, h, t, x, e))
        ts.append(t)
        xs.append(x)

        if not iflag:
            break

        if e < emin:
            h *= 2
        elif e > emax:
            h /= 2
            x = xsave
            t = tsave
            k -= 1

    plt.subplot(1, 2, 1)
    plt.plot(ts, xs)  # solved
    plt.grid()
    # plt.legend(["Solved"])
    plt.title("Solved function")

    yy = [theory_f(x) for x in ts]
    plt.subplot(1, 2, 2)
    plt.plot(ts, yy, c="red")  # expected
    # plt.legend(["Expected"])
    plt.title("Expected function")
    plt.grid()

    plt.show()
    print("[solved]   x({}) = {}".format(tb, xs[-1]))
    print("[expected] x({}) = {}".format(tb, theory_f(tb)))
    print("error: {}".format(np.fabs(xs[-1] - theory_f(tb))))
    plt.plot(ts, [np.fabs(i - j) for i, j in zip(xs, yy)])
    # plt.legend(["Error"])
    plt.title("Error")
    plt.grid()
    plt.show()


def task2():
    def f(t, x):
        return (3 * x / t) + (4.5 * t) - 13

    def expected_f(x):
        return np.power(x, 3) - (4.5 * np.power(x, 2)) + (6.5 * x)

    t = 3.0
    x = 6.0
    h = -1.0e-2
    tb = 0.5
    eps = 1.0e-10

    rk45ad(f, t, x, h, tb, eps, eps, expected_f)


if __name__ == "__main__":
    task1()
    task2()
    pass
