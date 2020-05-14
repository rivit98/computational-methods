import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import random
import matplotlib as mtplt

np.set_printoptions(precision=3)

def gen_graph(n, prob=0.5):
    while True:
        G = nx.binomial_graph(n, prob, None, True)
        if nx.number_strongly_connected_components(G) == 1:
            break

    return G


def draw_graph(G, layout, title=""):
    plt.figure(title)
    nx.draw_networkx_nodes(G, layout, node_color='blue', node_size=400)

    nx.draw_networkx_edges(G, layout, connectionstyle='arc3, rad = 0.2', width=1.3, alpha=0.5)
    nx.draw_networkx_labels(G, layout, font_size=10, font_color="white", font_weight='bold')

    plt.axis('off')
    plt.title(title)
    plt.draw()


class StatisticMethod:
    def __init__(self, A):
        self.A = A
        self.n = A.shape[0]

    def compute(self, single_run_len=150, total_runs=2000):
        statistics = np.zeros(self.n)
        for _ in range(total_runs):
            last_page = self.last_page(random.randrange(self.n), single_run_len)
            statistics[last_page] += 1

        statistics /= sum(statistics)
        return statistics

    def last_page(self, start_page, single_run_len):
        last_page_in_run = start_page
        for _ in range(single_run_len):
            last_page_in_run = self.next_page(self.A[last_page_in_run])

        return last_page_in_run

    def next_page(self, row_of_pobabilities):
        return np.random.choice(
            [i for i in range(self.n)],
            p=row_of_pobabilities
        )


class MatrixPowerMethod:
    def __init__(self, A):
        self.A = A
        self.n = A.shape[0]

    def compute(self, exponent=50, start_vec=None):
        if not start_vec:
            start_vec = np.zeros(self.n)
            start_vec[random.randrange(self.n)] = 1

        powered_matrix = np.linalg.matrix_power(self.A, exponent)
        result_vec = powered_matrix.T @ start_vec
        return result_vec


class EigenVectorMethod:
    def __init__(self, A, max_iterations=10 ** 4, epsilon=1e-6):
        self.A = A.T
        self.n = A.shape[0]
        self.max_iterations = max_iterations
        self.eps = epsilon

    def compute(self):
        eigen_vector = self.power_method()
        eigen_vector /= sum(eigen_vector)
        return eigen_vector

    def power_method(self):
        x_ = np.random.random(self.n)
        x_ /= np.linalg.norm(x_)
        for _ in range(self.max_iterations):
            x_i = self.A @ x_
            x_i /= np.linalg.norm(x_i)  # vec normalization
            if np.linalg.norm(x_i - x_) < self.eps:
                break
            x_ = x_i

        return x_.T


def random_surfer(G):
    print("Graph with {} nodes".format(G.number_of_nodes()))
    draw_graph(G, nx.spring_layout(G))
    plt.show()

    adj_matrix = nx.to_numpy_array(G)
    normalized_adj_matrix = adj_matrix / adj_matrix.sum(axis=1)[:, None]  # normalize

    # print(adj_matrix)
    # print(normalized_adj_matrix)

    eigen_vector_result = EigenVectorMethod(normalized_adj_matrix).compute()
    # print("Eigen vector method:", eigen_vector_result)

    matrix_power_method_result = MatrixPowerMethod(normalized_adj_matrix).compute()
    # print("Matrix power method:", matrix_power_method_result)

    statistical_result = StatisticMethod(normalized_adj_matrix).compute()
    # print("Statistical method:", statistical_result)

    # print("Verification:")
    # print("> Eigen vector method sum: {}".format(sum(eigen_vector_result)))
    # print("> Matrix power method: {}".format(sum(matrix_power_method_result)))
    # print("> Statistical method: {}".format(sum(statistical_result)))

    fmt = "|{:^.4f}|{:^.4f}|{:^.4f}|"
    print("|{:^6}|{:^6}|{:^6}|".format("EIGEN", "MAT", "STAT"))
    print('-' * 22)
    for i in range(len(eigen_vector_result)):
        print(fmt.format(eigen_vector_result[i], matrix_power_method_result[i], statistical_result[i]))


def task1():
    # TEST GRAPH
    arr = np.array([[0, 0, 1, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 1, 0]])

    G = nx.from_numpy_array(arr, create_using=nx.DiGraph)
    ###########

    random_surfer(G)

    for n in [15, 25, 50]:
        G = gen_graph(n, 0.4)
        random_surfer(G)


def e_random(n):
    return np.random.random_sample(n).reshape((n, 1))

def e_same_weight(n):
    return np.ones(n).reshape((n, 1))

def page_rank(A, d, e_func=e_random, eps=1e-6):
    n = A.shape[0]
    # A /= A.sum(axis=1)[:, None] # doesnt work because of zeroes

    e_vec = e_func(n)
    e_vec /= np.linalg.norm(e_vec, ord=1)
    B = np.array(d * A.T + ((1-d) * (e_vec @ np.ones(shape=(1, n)))), dtype=float)

    r = e_vec.copy()
    delta = 1
    while delta > eps:
        r_next = B @ r
        d_ = np.linalg.norm(r, ord=1) - np.linalg.norm(r_next, ord=1)
        r_next += (d_ * e_vec)
        delta = np.linalg.norm(r_next - r, ord=1)
        r = r_next

    r = np.array(r)
    print(r)
    print("Sum == 1? {}".format("Yes" if sum(r) else "No"))
    # print(r.reshape(-1))


def load_graph(filename):
    G = nx.DiGraph()
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if line[0] != '#':
                a, b = line.split()
                G.add_edge(a, b)

    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return nx.to_numpy_array(G)

def task2():
    # TEST GRAPH
    arr = np.array([[0, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                    [1, 0, 1, 0]])

    G = nx.from_numpy_array(arr, create_using=nx.DiGraph)
    # page_rank(nx.to_numpy_array(G), 0.85)

    data_format = "./data/{}"

    files = ["p2p-Gnutella09.txt", "Wiki-Vote.txt", "email-Eu-core.txt"]
    ds = [0.9, 0.85, 0.75, 0.6, 0.5]
    for f in files:
        A = load_graph(data_format.format(f))

        for i, row in enumerate(A):  # normalize only once
            s = sum(row)
            if s:
                A[i] = row / s

        for d in ds:
            for e in [e_random, e_same_weight]:
                print("Graph: {} ({} nodes), d={}, e={}".format(f, A.shape[0] ,d, e.__name__))

                page_rank(A, d, e)
                print("\n")


if __name__ == "__main__":
    # task1()
    task2()
    pass
