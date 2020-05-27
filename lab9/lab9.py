import numpy as np
import random
import matplotlib as mtplt
from matplotlib import pyplot as plt



def get_random_square_matrix(n):
    return np.random.rand(n, n)


def my_qr(A_orig):
    A = A_orig.copy()
    n = A.shape[0]
    Q = np.zeros(shape=A.shape, dtype=np.float64)
    R = np.zeros(shape=A.shape, dtype=np.float64)

    def norm_col(col):
        return col / np.linalg.norm(col)

    def back_sum(U, A, n):
        return sum((U[:, i].dot(A[:, n])) * U[:, i] for i in range(n))

    for k in range(0, n):
        Q[:, k] = norm_col(A[:, k] - back_sum(Q, A, k))

    for i in range(n):
        for j in range(i, n):
            R[i][j] = np.dot(Q[:, i], A[:, j])

    return Q, R


def task1():

    def compare_matrices(A, B, eps=1e-6):
        return np.allclose(np.abs(A), np.abs(B), eps)

    format_str = "|{:^5}|{:^17}|"
    print(format_str.format("n", "my_qr == linalg"))
    for i in range(5):
        n = random.randint(10, 200)
        A = get_random_square_matrix(n)
        qq, rr = my_qr(A)
        q, r = np.linalg.qr(A)

        equal = compare_matrices(qq, q) and compare_matrices(rr, r)
        print(format_str.format(n, "equal" if equal else "not equal"))


    # 3
    matrices_num = 80
    n = 8

    matrices = []
    conds = []
    for i in range(matrices_num):
        cond = random.randint(500, 2000)
        m1, m2, A = tuple(get_random_square_matrix(n) for _ in range(3))

        q1, _ = np.linalg.qr(m1)
        q2, _ = np.linalg.qr(m2)
        _, S, _ = np.linalg.svd(A)
        S[0] = cond * S[-1]

        matrices.append(q1 @ np.diag(S) @ q2)
        conds.append(cond)


    norms = []
    for matrix in matrices:
        Q, _ = my_qr(matrix)
        norms.append(np.linalg.norm(np.identity(n) - (Q.T @ Q)))

    plt.scatter(conds, norms, s=13, c="red")
    plt.xlabel("cond(A)")
    plt.ylabel("|| I - Q @ Q.T ||")
    plt.legend(["Error"])
    plt.title("Error depending on condition")
    plt.grid()
    plt.show()



def task2():
    X = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float64)
    Y = np.array([2, 7, 9, 12, 13, 14, 14, 13, 10, 8, 4], dtype=np.float64)
    A = np.array([[1, x, x**2] for x in X], dtype=np.float64)

    Q, R = np.linalg.qr(A)
    x = np.linalg.solve(R, Q.T @ Y)
    print(x)

    xs = np.linspace(-10, 10, 100)
    newY = list(map(lambda point: x[0] + (x[1] * point) + (x[2] * (point**2)), xs))

    plt.scatter(X, Y, s=20, c="red")
    plt.plot(X, newY, c="blue")
    plt.rc('axes', axisbelow=False)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Approximated function", "Original points"])
    plt.title("Approximated function")
    plt.show()




if __name__ == "__main__":
    # task1()
    # task2()
    pass