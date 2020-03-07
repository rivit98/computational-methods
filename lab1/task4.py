import numpy

from mylib import *


def log_func(r, x):
    return r * x * (1-x)


def bif_diag(x0, r_start, r_end, step, iterations, skip, float_type):
    print('x0: {} | iterations: {} | skip: {} | float_type: {}'.format(x0, iterations, skip, float_type.__name__))
    r = float_type(r_start)
    step = float_type(step)

    values = []
    xaxis = []
    while r <= r_end:
        x = float_type(x0)

        for i in range(iterations):
            x = float_type(log_func(r, x))
            if i > skip:
                values.append(x)
                xaxis.append(r)

        r += step

    plt.plot(xaxis, values, markersize=1, marker='.',  ls='')
    plt.show()


def subtask_c():
    r = float32(4)
    step = float(0.001)
    cap = 8000
    x0 = float32(0)
    res = []
    xaxis = []
    eps = float32(0.000001)
    while x0 <= 1:
        x = x0
        iterations = 0
        for i in range(cap):
            if x <= eps:
                print(x0)
                xaxis.append(x0)
                res.append(iterations)
                break

            x = float32(log_func(r, x))
            iterations += 1

        x0 += step

    plt.plot(xaxis, res, markersize=5, marker='.',  ls='')
    plt.show()


def task4():
    # x0 = [float32(0.2), float32(0.53), float32(0.96)]
    # x0_64 = [float64(0.2), float64(0.53), float64(0.96)]
    # make_header('a)')
    # for i in range(len(x0)):
    #     bif_diag(x0[i], 1.0, 4.0, 0.005, 200, 100, float32)
    #
    # make_header('b')
    # step = 0.00005
    # for i in range(len(x0)):
    #     bif_diag(x0[i], 3.75, 3.8, step, 140, 100, float32)
    #     bif_diag(x0_64[i], 3.75, 3.8, step, 140, 100, float64)

    make_header('c')
    subtask_c()


if __name__ == "__main__":
    task4()
