import numpy as np
from numpy.random import default_rng
from scipy.special import gammaincc
import random
import functools
import matplotlib as mtplt
from scipy.stats import norm, shapiro
from matplotlib import pyplot as plt


def approximate_entropy_test(seq, m=2, n=3):
    def to_number_from_m_bits(arr):
        return np.packbits(arr, axis=-1) >> (8 - m)

    def to_number_from_m_plus_bits(arr):
        return np.packbits(arr, axis=-1) >> (7 - m)

    seq = list(map(lambda x: int(x), seq))
    seq_m = seq + seq[:m - 1]
    seq_m_plus = seq + seq[:m]
    C_m = np.zeros(to_number_from_m_bits(np.ones(m, dtype=np.int8)) + 1)
    C_m_plus = np.zeros(to_number_from_m_plus_bits(np.ones(m + 1, dtype=np.int8)) + 1)

    for i in range(n):
        C_m[to_number_from_m_bits(seq_m[i: i + m])] += 1
        C_m_plus[to_number_from_m_plus_bits(seq_m_plus[i: i + m + 1])] += 1

    C_m, C_m_plus = C_m / n, C_m_plus / n
    phi_m = 0
    for c in C_m:
        if c != 0:
            phi_m += c * np.log(c)

    phi_m_plus = 0
    for c in C_m_plus:
        if c != 0:
            phi_m_plus += c * np.log(c)

    ap_en = phi_m - phi_m_plus
    xsi = 2 * n * (np.log(2) - ap_en)
    p_value = gammaincc(2 ** (m - 1), xsi / 2)

    print(p_value)
    return p_value >= 0.01


def test_generator(n, generator, title):
    BUCKETS_NUM = 10
    rand_nums = generator(n)
    prev = sum(j > i for i, j in zip(rand_nums, rand_nums[1:]))
    test_result = approximate_entropy_test(rand_nums)

    plt.hist(rand_nums, BUCKETS_NUM, facecolor='blue', alpha=0.5, ec='black')
    plt.title("{} - {} samples".format(title, n))
    plt.ylabel("Number of occurrences")
    plt.xlabel("x")
    plt.show()

    print("[{}] x_i < x_(i+1) is satisfied for {} numbers ({}%)".format(title, prev, round(prev * 100 / n)))
    print("Approximate Entropy Test: Sequence is {}\n\n"
          .format("random" if test_result else "not random"))


def task1():
    INTERVAL_BEGIN = 0
    INTERVAL_END = 100
    n = [10, 1000, 5000]

    gens = [
        ("Mersenne Twister",
         functools.partial(lambda x: [random.randint(INTERVAL_BEGIN, INTERVAL_END) for _ in range(x)])),
        ("PCG64", functools.partial(default_rng().uniform, INTERVAL_BEGIN, INTERVAL_END))
    ]

    for title, gen in gens:
        for nn in n:
            test_generator(nn, gen, title)


class BoxMullerGenerator:
    def __init__(self, mu=0, sig=1):
        self.sigma = sig  # standard deviation
        self.mu = mu  # expected value
        self.generateNewPair = False
        self.x1 = 0
        self.x2 = 0

    def get_sigma(self):
        return self.sigma

    def get_mu(self):
        return self.mu

    def compute_return(self, val):
        return (val * self.sigma) + self.mu

    def generate(self):
        self.generateNewPair = not self.generateNewPair

        if not self.generateNewPair:
            return self.compute_return(self.x2)

        u1, u2 = random.random(), random.random()
        self.x1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        self.x2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        return self.compute_return(self.x1)


def test_generator_box_muller(n, mu=0, sig=1):
    gen = BoxMullerGenerator(mu, sig)
    nums = [gen.generate() for _ in range(n)]

    plt.hist(nums, 10, facecolor='blue', alpha=0.5, ec='black', density=True)
    plt.title(r"Box-Muller generator - {} samples $(\mu={}, \sigma={})$".format(n, mu, sig))
    plt.ylabel("Frequency of occurrences")
    plt.xlabel("x")

    x_axis = np.linspace(min(nums)-5, max(nums)+5, 1000)
    plt.plot(x_axis, norm.pdf(x_axis, mu, sig), c='red', linestyle='--')

    plt.show()

    _, p = shapiro(nums)
    alpha = 0.05
    if p > alpha:
        print("Hypothesis zero accepted - data is from a normal distribution")
    else:
        print("Hypothesis zero rejected - data is NOT from a normal distribution")

    print("\n\n")


def task2():
    n = [10, 1000, 5000]

    for nn in n:
        test_generator_box_muller(nn)

    test_generator_box_muller(5000, 50, 15)



def estimate_pi(n):
    x = np.random.uniform(-1.0, 1.0, n)
    y = np.random.uniform(-1.0, 1.0, n)
    d = np.square(x) + np.square(y)
    q = (d <= 1)  # n-element list with booleans indicating if point is in circle or not

    if n < 30000 and not n % 3000:
        bounds = [-1.0, 1.0]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(x[q], y[q], '.', color='green')
        ax.plot(x[np.logical_not(q)], y[np.logical_not(q)], '.', color='red')

        circ = plt.Circle((0.0, 0.0), radius=1, color='gray', fill=False, label='circle')
        ax.add_patch(circ)
        ax.set_aspect("equal")

        plt.title(r"Monte Carlo: $\pi$ estimation ({} points)".format(n))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(bounds)
        plt.ylim(bounds)
        plt.show()

    return 4 * (q.sum() / len(q))


def task3():
    n = [i*500 for i in range(1, 500)]
    computed_pi = [estimate_pi(nn) for i, nn in enumerate(n)]
    errors = list(map(lambda x: np.fabs(x - np.pi), computed_pi))

    plt.plot(n, errors)
    plt.title(r"$\pi$ estimation error")
    plt.ylabel("error")
    plt.xlabel("number of points")
    plt.show()

    print("np.pi              : {}".format(np.pi))
    print("best monte carlo pi: {}".format(computed_pi[errors.index(min(errors))]))
    print("error              : {}".format(min(errors)))



if __name__ == "__main__":
    task1()
    task2()
    task3()
    pass
