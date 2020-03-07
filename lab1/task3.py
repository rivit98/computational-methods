from mylib import *


def riemman_func(k, s, float_type):
    return float_type(1 / (k ** s))


def dirichlet_func(k, s, float_type):
    return float_type(((-1) ** (k - 1)) * riemman_func(k, s, float_type))


def sum_riemann(n, s, float_type, dir=1):
    start = 1 if dir == 1 else n+1
    stop = n + 1 if dir == 1 else 0

    sum = float_type(0.0)
    for k in range(start, stop, dir):
        sum += riemman_func(k, s, float_type)

    return sum


def sum_dirichlet(n, s, float_type, dir=1):
    start = 1 if dir == 1 else n+1
    stop = n + 1 if dir == 1 else 0

    sum = float_type(0.0)
    for k in range(start, stop, dir):
        sum += dirichlet_func(k, s, float_type)

    return sum


def compare_realtive_errors_dirichlet(n, s, float_type, dir=1):
    print('Dirichlet | step: {} | iter: {} | float_type: {}'.format(s, n, float_type.__name__))
    err1 = []
    err2 = []
    start = 1 if dir == 1 else n+1
    stop = n + 1 if dir == 1 else 0

    for k in range(start, stop, dir):
        x = dirichlet_func(k, s, float_type)
        y = dirichlet_func(k+1, s, float_type)
        z = dirichlet_func(k+2, s, float_type)
        err1.append(relative_error(float_type(x + y + z), float_type(float_type(x+y) + z)))
        err2.append(relative_error(float_type(x + y + z), float_type(x + float_type(y+z))))

    print(err1)
    print(err2)

    plt.plot(err1)
    plt.show()
    plt.plot(err2)
    plt.show()


def compare_realtive_errors_riemann(n, s, float_type, dir=1):
    print('Riemann | step: {} | iter: {} | float_type: {}'.format(s, n, float_type.__name__))
    err1 = []
    err2 = []
    start = 1 if dir == 1 else n+1
    stop = n + 1 if dir == 1 else 0

    for k in range(start, stop, dir):
        x = riemman_func(k, s, float_type)
        y = riemman_func(k+1, s, float_type)
        z = riemman_func(k+2, s, float_type)
        err1.append(relative_error(float_type(x + y + z), float_type(float_type(x+y) + z)))
        err2.append(relative_error(float_type(x + y + z), float_type(x + float_type(y+z))))

    print(err1)
    print(err2)

    plt.plot(err1)
    plt.show()
    plt.plot(err2)
    plt.show()


def task3():
    s32 = [float32(2), float32(3.6667), float32(5), float32(7.2), float32(10)]
    s64 = [float64(2), float64(3.6667), float64(5), float64(7.2), float64(10)]
    n = [50, 100, 200, 500, 1000]
    dirs = [1, -1]

    for sum_function in [sum_dirichlet, sum_riemann]:
        print('-' * 120)
        print('|{:^118}|'.format(sum_function.__name__))
        print('-' * 120)
        for i in range(len(s32)):
            print('|{:^20}|{:^13}|{:^20}|{:^20}|{:^20}|{:^20}|'
                  .format('Step', 'Iterations', 'Single forward',  'Single backward', 'Double forward', 'Double backward'))
            for iterations in n:
                result = []
                for precision in [float32, float64]:
                    for dir in dirs:
                        step_to_use = s32[i] if precision == float32 else s64[i]
                        result.append(sum_function(iterations, step_to_use, precision, dir))

                print('|{:^20}|{:^13}|{:^20}|{:^20}|{:^20}|{:^20}|'.format(step_to_use, iterations, result[0], result[1],
                                                                           result[2], result[3]))
            print('-' * 120)
        print('\n')

    # pare wnioskow
    # w obu funkcjach przy kroku >= 7.2 roznice w wynikach sa prawie niezauwazalne
    # dla szeregu naprzemiennym (dirichlet) wyniki sa bardziej zblizone do siebie, dzieki temu ze dodawane do siebie liczby w kolejnych krokach
    # sa mniejsze niz w przypadku zwyklego dodawania

    compare_realtive_errors_dirichlet(50, float32(2.0), float32, 1)
    compare_realtive_errors_dirichlet(50, float64(2.0), float64, 1)

    compare_realtive_errors_riemann(50, float32(2.0), float32, 1)
    compare_realtive_errors_riemann(50, float64(2.0), float64, 1)



if __name__ == "__main__":
    task3()
