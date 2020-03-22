import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

np.set_printoptions(precision=3)


def draw_3d_plot(x, y, z, view=(10, 20)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=view[0], azim=view[1])
    ax.plot_surface(x, y, z, alpha=0.4)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


# 1.1
def task1_1():
    s = np.linspace(0, 2 * np.pi, 50)
    t = np.linspace(0, np.pi, 50)

    x = np.outer(np.cos(s), np.sin(t))
    y = np.outer(np.sin(s), np.sin(t))
    z = np.outer(np.ones(np.size(s)), np.cos(t))

    draw_3d_plot(x, y, z)


# 1.2
def transform_sphere(A):
    s = np.linspace(0, 2 * np.pi, 50)
    t = np.linspace(0, np.pi, 50)
    x = (np.outer(np.cos(s), np.sin(t)) * A.item(0, 0)) + \
        (np.outer(np.sin(s), np.sin(t)) * A.item(0, 1)) + \
        (np.outer(np.ones(np.size(s)), np.cos(t)) * A.item(0, 2))

    y = (np.outer(np.cos(s), np.sin(t)) * A.item(1, 0)) + \
        (np.outer(np.sin(s), np.sin(t)) * A.item(1, 1)) + \
        (np.outer(np.ones(np.size(s)), np.cos(t)) * A.item(1, 2))

    z = (np.outer(np.cos(s), np.sin(t)) * A.item(2, 0)) + \
        (np.outer(np.sin(s), np.sin(t)) * A.item(2, 1)) + \
        (np.outer(np.ones(np.size(s)), np.cos(t)) * A.item(2, 2))

    return x, y, z


A = []
A.append(np.array([[1, 2, 0], [1, 1, 1], [3, 1, 0]]))
A.append(np.array([[4, 0, 4], [0, 3, 0], [-3, 0, 3]]))
A.append(np.array([[5, 0, 1], [2, 3, 0], [0, 0, 4]]))


def task1_2():
    for transform_matrix in A:
        x1, y1, z1 = transform_sphere(transform_matrix)
        draw_3d_plot(x1, y1, z1)


# 1.3

def task1_3():
    for transform_matrix in A:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=10, azim=20)

        U, S, VT = np.linalg.svd(transform_matrix)
        S_diag = np.diag(S)  # because S is vector and we need diagonal matrix
        x1, y1, z1 = transform_sphere(transform_matrix)

        for s1_row in S_diag:
            base = np.dot(U, s1_row)  # 3d base vec for one axis
            ax.plot([0, base[0]], [0, base[1]], [0, base[2]], linewidth=4)

        ax.plot_surface(x1, y1, z1, alpha=0.3, color='g')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()


# 1.4

def get_matirx_with_ratio_greater_than_100():
    while True:
        AA = np.random.rand(3, 3)
        U, S, VT = np.linalg.svd(AA)

        if S[0] / S[2] > 100:
            return AA


def task1_4():
    A100 = get_matirx_with_ratio_greater_than_100()
    U, S, VT = np.linalg.svd(A100)
    print("Znaleziona macierz:")
    print(A100)  # found matrix
    print("\nWartości osobliwe tej macierzy")
    print(np.diag(S))  # singular values
    print("\nStosunek najwiekszej wartosci osobliwej do najmniejszej: {}".format(S[0] / S[2]))
    x1, y1, z1 = transform_sphere(A100)
    draw_3d_plot(x1, y1, z1)
    draw_3d_plot(x1, y1, z1, (10, 90))  # ellipses might be different, so generate three views
    draw_3d_plot(x1, y1, z1, (90, 80))


# 1.5
def check_one_limit(x, tuple_limits):
    a, b = abs(x.min()), abs(x.max())
    a = max(a, b)
    if a > tuple_limits[0]:
        return a

    return -1


def update_limits(ax, x, y, z, force=False):
    max_dim = max(check_one_limit(x, ax.get_xlim()), check_one_limit(y, ax.get_ylim()),
                  check_one_limit(z, ax.get_zlim()))

    if max_dim != -1 or force:
        ax.set_xlim(max_dim, -max_dim)
        ax.set_ylim(max_dim, -max_dim)
        # ax.set_zlim(max_dim, -max_dim)


def transform_sphere_and_draw_vectors(A, ax):
    # translate canonical vector
    ident = np.identity(3)
    for v in range(3):
        res_v = A @ ident[v, :]
        ax.quiver(0, 0, 0, res_v[0], res_v[1], res_v[2], color='b', arrow_length_ratio=0.2)
        # ax.plot([0, res_v[0]], [0, res_v[1]], [0, res_v[2]], linewidth=2, color='b')

    return transform_sphere(A)


def visualize_transform(matrix, alpha=0.3, rstride=5, cstride=5):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=30, azim=35)

    # base sphere
    x, y, z = transform_sphere(np.identity(3))
    ax.plot_wireframe(x, y, z, alpha=alpha, color='g', rstride=rstride, cstride=cstride)
    update_limits(ax, x, y, z)

    x, y, z = transform_sphere_and_draw_vectors(matrix, ax)
    ax.plot_wireframe(x, y, z, alpha=alpha, color='r', rstride=rstride, cstride=cstride)
    update_limits(ax, x, y, z)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def task1_5():
    M = A[0]  # our matrix
    U, S, VT = np.linalg.svd(M)
    S = np.diag(S)

    visualize_transform(VT)
    visualize_transform(S.dot(VT))
    visualize_transform(U.dot(S).dot(VT))
    visualize_transform(M)


def task2_2_grayscale():
    filename = "./img.jpg"
    img = Image.open(filename).convert('L')
    A = np.array(img)
    U, S, VT = np.linalg.svd(A)

    per_row = 3
    iter_num = 39
    for k in range(min(A.shape[0], iter_num)):
        I = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

        if k % per_row == 0 and k:
            plt.show()

        plt.subplot(1, per_row, (k % per_row) + 1)
        plt.title('k={}'.format(k))
        plt.axis('off')
        plt.imshow(I, cmap='gray')

    plt.show()


def task2_2_color():
    filename = "./img.jpg"
    img = Image.open(filename)
    w, h = img.width, img.height
    A = np.array(img)
    U, S, VT, I_colors = [None] * 3, [None] * 3, [None] * 3, [None] * 3
    for i in range(3):
        U[i], S[i], VT[i] = np.linalg.svd(A[:, :, i])

    per_row = 3
    iter_num = 39

    for k in range(min(A.shape[0], iter_num)):
        for i in range(3):
            # 0 - r, 1 - g, 2 - b
            I_colors[i] = U[i][:, :k] @ np.diag(S[i][:k]) @ VT[i][:k, :]
            I_colors[i].reshape(w * h)

        I = np.asarray([I_colors[0], I_colors[1], I_colors[2]]).transpose((1, 2, 0)).reshape(h, w, 3)
        img2 = Image.fromarray(np.uint8(I))

        if k % per_row == 0 and k:
            plt.show()
            pass

        plt.subplot(1, per_row, (k % per_row) + 1)
        plt.title('k={}'.format(k))
        plt.axis('off')
        plt.imshow(img2)

    plt.show()


def task2_3_grayscale():
    filename = "./img.jpg"
    img = Image.open(filename).convert('L')
    A = np.array(img)
    U, S, VT = np.linalg.svd(A)

    ranks = [1, 10, 20, 30, 40, 60, 100, 200]
    for i in range(min(A.shape[0], len(ranks))):
        k = ranks[i]
        I = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

        plt.subplot(1, 2, 1)
        plt.title('k={}'.format(k))
        plt.axis('off')
        plt.imshow(I, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('original')
        plt.axis('off')
        plt.imshow(img, cmap='gray')

        plt.show()


def task2_3_color():
    filename = "./img.jpg"
    img = Image.open(filename)
    w, h = img.width, img.height
    A = np.array(img)
    U, S, VT, I_colors = [None] * 3, [None] * 3, [None] * 3, [None] * 3
    for i in range(3):
        U[i], S[i], VT[i] = np.linalg.svd(A[:, :, i])

    ranks = [1, 10, 20, 30, 40, 60, 100, 200, 400]
    for ii in range(min(A.shape[0], len(ranks))):
        k = ranks[ii]
        for i in range(3):
            # 0 - r, 1 - g, 2 - b
            I_colors[i] = U[i][:, :k] @ np.diag(S[i][:k]) @ VT[i][:k, :]
            I_colors[i].reshape(w * h)

        I = np.asarray([I_colors[0], I_colors[1], I_colors[2]]).transpose((1, 2, 0)).reshape(h, w, 3)
        img2 = Image.fromarray(np.uint8(I))

        plt.subplot(1, 2, 1)
        plt.title('k={}'.format(k))
        plt.axis('off')
        plt.imshow(img2)

        plt.subplot(1, 2, 2)
        plt.title('original')
        plt.axis('off')
        plt.imshow(img)

        plt.show()


if __name__ == "__main__":
    # uncomment to run task
    task1_1()
    task1_2()
    task1_3()
    task1_4()
    task1_5()
    task2_2_grayscale()
    task2_2_color()
    task2_3_grayscale()
    task2_3_color()
    pass