import numpy as np
import matplotlib as mtplt
from matplotlib import pyplot as plt
import random
import time


np.set_printoptions(precision=3)

def get_random_vec(r=None):
    if not r:
        r = random.randint(1, 10)
    n = 2 ** r
    return np.random.uniform(low=-10, high=10, size=n)


def fourier_matrix(n):
    def dzeta(j, k):
        return np.exp(((-2j) * np.pi) / n) ** (j * k)

    return np.array([dzeta(j, k) for j in range(n) for k in range(n)], dtype=np.complex).reshape((n, n))


def dft(x):
    n = x.shape[0]
    F = fourier_matrix(n)

    return F @ x, F


def idft(F, y):
    return (np.conjugate(F @ np.conjugate(y))) / y.shape[0]


def fft_recursive(x):
    n = x.shape[0]

    if n == 1:
        return x[0]

    even_x = fft_recursive(x[::2])
    odd_x = fft_recursive(x[1::2])
    f = np.exp((np.pi * (-2j) * np.arange(n)) / n)
    return np.concatenate([even_x + f[:n//2] * odd_x, even_x + f[n//2:] * odd_x])


def timeit(func):
    startTime = time.perf_counter()
    return func(), round(time.perf_counter() - startTime, 4)


def task1():
    for _ in range(10):
        x = get_random_vec()
        y, F = dft(x)
        x_ = idft(F, y)
        print("(dft == numpy fft)? ", end='')
        print(np.allclose(y, np.fft.fft(x), rtol=1e-6) and np.allclose(np.real(x_), x, rtol=1e-6))

    for _ in range(10):
        x = get_random_vec()
        print("(fft_recursive == numpy fft)? ", end='')
        print(np.allclose(fft_recursive(x), np.fft.fft(x), rtol=1e-6))

    format_str = "|{:^11}|{:^15}|{:^15}|{:^15}|{:^8}|"
    print(format_str.format("Vec size", "DFT", "Cooley-Turkey", "numpy fft", "Valid?"))
    for r in range(1, 12):
        x = get_random_vec(r)
        res1, time1 = timeit(lambda: dft(x)[0])
        res2, time2 = timeit(lambda: fft_recursive(x))
        res3, time3 = timeit(lambda: np.fft.fft(x))

        valid = np.allclose(res1, res2) and np.allclose(res2, res3)

        print(format_str.format(2 ** r, time1, time2, time3, "True" if valid else "False"))



DATA_SIZE = 512
def get_signals(n):
    xs = np.linspace(0, 50, DATA_SIZE)
    ys_ = []

    for i in range(1, n+1):
        ys = np.sin(xs * i)
        ys_.append(ys)

    return xs, ys_


def plot_signals(xs, ys):
    plt.subplots_adjust(hspace=1)
    for i, y in enumerate(ys):
        plt.subplot(5, 1, i+1)
        plt.title("sin({}π)".format(i+1))
        plt.plot(xs, y)

    plt.show()


def get_overlapping_signal(ys):
    return [sum(i) for i in zip(*ys)]


def get_concatenated_signal(ys):
    res = []
    divider = DATA_SIZE / 5
    for i, v in enumerate(ys):
        res.extend(v[round(i*divider):round((i+1)*divider)])

    return res


def plots_two_signals(xs, overlapped_signal, concatenated_signal):
    plt.plot(xs, overlapped_signal)
    plt.title("Nałożone na siebie sygnały sinusoidalne o różnych częstotliwościach")
    plt.show()

    plt.title("Złączone ze sobą sygnały sinusoidalne o różnych częstotliwościach")
    plt.plot(xs, concatenated_signal)
    plt.show()


def analyze_freq(xs, data):
    y = np.fft.fft(data)

    plt.plot(xs, np.real(y))
    plt.title("Część rzeczywista")
    plt.show()

    plt.plot(xs, np.imag(y))
    plt.title("Część urojona")
    plt.show()


def task2():
    xs, ys = get_signals(5)
    plot_signals(xs, ys)

    overlapped_signal = get_overlapping_signal(ys)
    concatenated_signal = get_concatenated_signal(ys)

    plots_two_signals(xs, overlapped_signal, concatenated_signal)

    analyze_freq(xs, overlapped_signal)
    analyze_freq(xs, concatenated_signal)



if __name__ == "__main__":
    # task1()
    task2()
    pass