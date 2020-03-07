from numpy import float32, float64
from matplotlib import pyplot as plt
import time


def make_header(text):
    s = '*' * 20
    print('{} {} {}'.format(s, text, s))


def printTime(startTime):
    print('{:<20}: {}s'.format('Czas', round(time.perf_counter() - startTime, 4)))


def printExpected(expected):
    print('{:<20}: {}'.format('Oczekiwana', expected))


def printCalculated(calculated):
    print('{:<20}: {}'.format('Obliczona', calculated))


def printErrors(expected, calculated):
    print('{:<20}: {}'.format('Bezwzgledny', absolute_error(expected, calculated)))
    print('{:<20}: {}'.format('Wzgledny', relative_error(expected, calculated)))


def printData(expected, calculated):
    printExpected(expected)
    printCalculated(calculated)
    printErrors(expected, calculated)


def relative_error(expected, current):
    return float64(absolute_error(expected, current) / expected)


def blad_wzgledny_percent(expected, current):
    return relative_error(expected, current) * 100


def absolute_error(expected, current):
    return float64(abs(expected - current))


def suma_float32(arr):
    suma_ = float32(0)
    for i in arr:
        suma_ += i

    return suma_
