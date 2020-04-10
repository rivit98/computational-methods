from mpmath import *
import inspect
from matplotlib import pyplot as plt
import numpy as np

functions = [
    {
        "func": lambda x: (cos(x) * cosh(x)) - 1,
        "func_der": lambda x: (cos(x) * sinh(x)) - (cosh(x) * sin(x)),
        "left_bound": (3 / 2) * pi,
        "right_bound": 2 * pi,
    },
    {
        "func": lambda x: (1 / x) * tan(x),
        "func_der": lambda x: ((x * power(sec(x), 2)) - tan(x)) / power(x, 2),
        "left_bound": 0,
        "right_bound": pi / 2,
    },
    {
        "func": lambda x: power(2, -x) + power(e, x) + 2 * cos(x) - 6,
        "func_der": lambda x: power(e, x) - (power(2, -x) * log(2)) - (2 * sin(x)),
        "left_bound": 1,
        "right_bound": 3,
    }
]


def get_max_iter(a, b, eps):
    return int(ceil(log((b - a) / eps) / log(2)))


def safe_calc(f, a):
    try:
        return mpf(f(a))
    except ZeroDivisionError:
        return inf


def bisection(func, eps, precision, silent=False):
    mp.dps = precision
    f, a, b = func['func'], func['left_bound'], func['right_bound']
    fa, fb = safe_calc(f, a), safe_calc(f, b)

    iter = 0
    error = mpf(b - a)

    if fa == 0:
        return a, f(a), iter
    elif fb == 0:
        return b, f(b), iter
    elif fa * fb > 0:  # same signs
        if not silent:
            print("Function does not have root in range [{};{}]".format(a, b))
        return None
    else:
        for n in range(get_max_iter(a, b, eps)):
            iter += 1
            error /= 2
            c = a + error
            fc = f(c)
            if fabs(error) < eps:
                return c, f(c), iter

            if fa * fc >= 0:
                a = c
                fa = fc


def get_k_roots(f, k):
    roots = []
    i = 0
    prec, eps = 15, mpf(1e-15)
    f_dict = {
        "func": f,
        "left_bound": 0,
        "right_bound": 2
    }
    while i < k:
        res = bisection(f_dict, eps, prec, True)
        if res is not None:
            roots.append(res[0])
            i += 1

        f_dict['left_bound'] = f_dict['right_bound']
        f_dict['right_bound'] += 1

    return roots


def task1():
    bound_correct = 0.1
    print("1) Bisection method")
    epsilon = [mpf(1e-7), mpf(1e-15), mpf(1e-33)]
    precision = [7, 15, 33]
    for f in functions:
        fname = inspect.getsourcelines(f['func'])[0][0].split(':')[2].strip()[:-1]
        print("\nFunction: " + fname)
        ret = None
        for i in range(len(precision)):
            print("\nPrecision: {}".format(precision[i]))
            ret = bisection(f, epsilon[i], precision[i])
            if ret is not None:
                print("Root at {}, value {}, iterations {}".format(ret[0], ret[1], ret[2]))

        x = np.linspace(np.float32(f['left_bound']) - bound_correct, np.float32(f['right_bound']) + bound_correct, 100)
        plt.plot(x, [f['func'](_x) for _x in x], label=fname)
        plt.axvline(np.float32(f['left_bound']), color="green", label="range", linestyle="--")
        plt.axvline(np.float32(f['right_bound']), color="green", linestyle="--")
        if ret is not None:
            plt.plot([ret[0]], [f['func'](ret[0])], marker='o', markersize=5, color="red")
        plt.legend()
        plt.grid()
        plt.show()

    print('*' * 50)
    k = 10
    f = functions[0]
    roots = get_k_roots(f['func'], k)
    x = np.linspace(0 - bound_correct, np.float32(roots[-1]) + bound_correct, 10000)
    plt.plot(x, [f['func'](_x) for _x in x])
    for r in roots:
        plt.plot([r], [f['func'](r)], marker='o', markersize=5, color="red")
    plt.ylim(-50, 50)
    plt.grid()
    # plt.show()



def newton(func, eps, precision):
    mp.dps = precision
    f, x = func['func'], func['left_bound']
    fx = safe_calc(f, x)
    delta = mpf(1e-20)

    for i in range(get_max_iter(func['left_bound'], func['right_bound'], eps)):
        fp = mpf(safe_calc(func['func_der'], x))
        if fabs(fp) < delta:
            return None

        d = mpf(fx/fp)
        x -= d
        fx = f(x)
        if fabs(d) < eps:
            return x, f(x), i

    print("No root found. Max steps reached")


def task2():
    bound_correct = 0.1
    print("2) Newton method")
    epsilon = [mpf(1e-7), mpf(1e-15), mpf(1e-33)]
    precision = [7, 15, 33]
    for f in functions:
        fname = inspect.getsourcelines(f['func'])[0][0].split(':')[2].strip()[:-1]
        print("\nFunction: " + fname)
        ret = None
        for i in range(len(precision)):
            print("\nPrecision: {}".format(precision[i]))
            ret = newton(f, epsilon[i], precision[i])
            if ret is not None:
                print("Root at {}, value {}, iterations {}".format(ret[0], ret[1], ret[2]))

        x = np.linspace(np.float32(f['left_bound']) - bound_correct, np.float32(f['right_bound']) + bound_correct, 100)
        plt.plot(x, [f['func'](_x) for _x in x], label=fname)
        plt.axvline(np.float32(f['left_bound']), color="green", label="range", linestyle="--")
        plt.axvline(np.float32(f['right_bound']), color="green", linestyle="--")
        if ret is not None:
            plt.plot([ret[0]], [f['func'](ret[0])], marker='o', markersize=5, color="red")
        plt.legend()
        plt.grid()
        plt.show()


def secant(func, eps, precision):
    mp.dps = precision
    f, a, b = func['func'], func['left_bound'], func['right_bound']
    fa, fb = safe_calc(f, a), safe_calc(f, b)

    if fa * fb > 0:  # same signs
        print("Function does not have root in range [{};{}]".format(a, b))
        return None

    if fabs(fa) > fabs(fb):
        a, b = b, a
        fa, fb = fb, fa

    for i in range(2, get_max_iter(func['left_bound'], func['right_bound'], eps), 1):
        if fabs(fa) > fabs(fb):
            a, b = b, a
            fa, fb = fb, fa

        d = mpf((b - a) / (fb - fa))
        b = a
        fb = fa
        d *= fa
        if fabs(d) < eps:
            return a, f(a), i

        a -= d
        fa = f(a)


    print("No root found. Max steps reached")


def task3():
    bound_correct = 0.1
    print("3) Secant method")
    epsilon = [mpf(1e-7), mpf(1e-15), mpf(1e-33)]
    precision = [7, 15, 33]
    for f in functions:
        fname = inspect.getsourcelines(f['func'])[0][0].split(':')[2].strip()[:-1]
        print("\nFunction: " + fname)
        ret = None
        for i in range(len(precision)):
            print("\nPrecision: {}".format(precision[i]))
            ret = secant(f, epsilon[i], precision[i])
            if ret is not None:
                print("Root at {}, value {}, iterations {}".format(ret[0], ret[1], ret[2]))

        x = np.linspace(np.float32(f['left_bound']) - bound_correct, np.float32(f['right_bound']) + bound_correct, 100)
        plt.plot(x, [f['func'](_x) for _x in x], label=fname)
        plt.axvline(np.float32(f['left_bound']), color="green", label="range", linestyle="--")
        plt.axvline(np.float32(f['right_bound']), color="green", linestyle="--")
        if ret is not None:
            plt.plot([ret[0]], [f['func'](ret[0])], marker='o', markersize=5, color="red")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # uncomment to run task
    # task1()
    # task2()
    # task3()
    pass

