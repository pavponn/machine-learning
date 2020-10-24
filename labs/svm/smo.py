import random
from kernels import Kernel, calc_kernel, calculate_kernels


class SVMModel(object):
    def __init__(self, n, c, xs, ys, kernel: Kernel, param):
        self.alphas = [0] * n
        self.b = 0
        self.c = c
        self.xs = xs
        self.ks = calculate_kernels(xs, kernel, param)
        self.ys = ys
        self.kernel = kernel
        self.param = param

    def predict(self, x):
        assert (len(self.xs[0]) == len(x))
        assert (len(self.xs) == len(self.alphas))
        res = 0
        for alpha, x_i, y_i in zip(self.alphas, self.xs, self.ys):
            res += alpha * y_i * calc_kernel(x_i, x, self.kernel, self.param)
        res += self.b
        if res > 0:
            return 1
        else:
            return -1


def f(model: SVMModel, ks, ys, i: int):
    n = len(ys)
    f_val = 0
    for j in range(n):
        f_val += model.alphas[j] * ks[i][j] * ys[j]
    f_val += model.b
    return f_val


def L_and_H(model: SVMModel, ys, i: int, j: int):
    if ys[i] != ys[j]:
        return max(0, model.alphas[j] - model.alphas[i]), min(model.c, model.c + model.alphas[j] - model.alphas[i])
    else:
        return max(0, model.alphas[j] + model.alphas[i] - model.c), min(model.c, model.alphas[j] + model.alphas[i])


def update_alpha(alpha_j, e_i, e_j, y_j: int, nu: float, ll: float, h: float):
    new_alpha = alpha_j - y_j * (e_i - e_j) / nu
    return put_alpha_in_range(new_alpha, ll, h)


def put_alpha_in_range(alpha_j, ll, h):
    if alpha_j > h:
        return h
    if h >= alpha_j >= ll:
        return alpha_j
    else:
        return ll


def simplified_SMO(xs, ys, c, kernel, param):
    eps = 1e-6
    tolerance = 1e-6
    n = len(ys)
    max_passes = 55000
    passes = 0

    cur_model = SVMModel(n, c, xs, ys, kernel, param)
    # Check all sizes
    assert (len(cur_model.ks) == len(ys))
    assert (len(xs) == len(ys))
    while passes < max_passes:
        others = list(range(n))
        random.shuffle(others)
        for i in range(n):
            passes += 1
            if passes > max_passes:
                break
            e_i = f(cur_model, cur_model.ks, ys, i) - ys[i]
            if (ys[i] * e_i < -tolerance and cur_model.alphas[i] < cur_model.c) or (
                    ys[i] * e_i > tolerance and cur_model.alphas[i] > 0):
                j = others[i]
                if j == i:
                    continue
                prev_alpha_i = cur_model.alphas[i]
                prev_alpha_j = cur_model.alphas[j]
                e_j = f(cur_model, cur_model.ks, ys, j) - ys[j]
                ll, h = L_and_H(cur_model, ys, i, j)
                if ll == h:
                    continue
                nu = 2 * cur_model.ks[i][j] - cur_model.ks[i][i] - cur_model.ks[j][j]
                if nu >= 0 or abs(nu) < 1e-7:
                    continue
                try:
                    new_alpha_j = update_alpha(prev_alpha_j, e_i, e_j, ys[j], nu, ll, h)
                except ZeroDivisionError:
                    continue
                cur_model.alphas[j] = new_alpha_j
                if abs(new_alpha_j - prev_alpha_j) < eps:
                    continue
                new_alpha_i = prev_alpha_i + ys[i] * ys[j] * (prev_alpha_j - new_alpha_j)
                cur_model.alphas[i] = new_alpha_i
                b1 = cur_model.b - e_i - ys[i] * (new_alpha_i - prev_alpha_i) * cur_model.ks[i][i] - ys[j] * (
                        new_alpha_j - prev_alpha_j) * cur_model.ks[i][j]
                b2 = cur_model.b - e_j - ys[i] * (new_alpha_i - prev_alpha_i) * cur_model.ks[i][j] - ys[j] * (
                        new_alpha_j - prev_alpha_j) * cur_model.ks[j][j]
                cur_model.b = (b1 + b2) / 2
                if 0 < new_alpha_i < cur_model.c:
                    cur_model.b = b1
                if 0 < new_alpha_j < cur_model.c:
                    cur_model.b = b2
    return cur_model
