from mylib import *


# mkromka@agh.edu.pl


def task1_1(x, numbers, expected):
    make_header('1 & 2')

    startTime = time.perf_counter()
    suma = suma_float32(numbers)
    printTime(startTime)

    printData(expected, suma)


def task1_3(x, numbers, expected):
    make_header('3')

    err = []
    partialSum = float32(0)
    for (index, value) in enumerate(numbers):
        partialSum += value

        if index % 25000 == 0 and index > 0:
            err.append(relative_error(index * x, partialSum))

    print('Done!')
    plt.plot(err)
    # plt.show()


def recur_sum(arr):
    if len(arr) == 1:
        return arr[0]

    len_ = int(len(arr) / 2)
    return recur_sum(arr[0:len_]) + recur_sum(arr[len_:])


def task1_4(x, numbers, expected):
    make_header('4')

    startTime = time.perf_counter()
    sum_ = recur_sum(numbers)
    printTime(startTime)
    printData(expected, sum_)


def task1():
    x = float32(0.53125)
    expected = 5312500
    numbers = [x] * (10 ** 7)

    # 1.1 1.2
    task1_1(x, numbers, expected)

    # 1.3
    task1_3(x, numbers, expected)

    # 1.4, 1.5
    task1_4(x, numbers, expected)

    # 1.7
    x = float32(0.0001)
    expected = 1000
    numbers = [x] * (10 ** 7)
    task1_4(x, numbers, expected)


if __name__ == "__main__":
    task1()
