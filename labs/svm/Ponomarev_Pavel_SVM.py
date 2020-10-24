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

from enum import Enum, auto
import math


class Kernel(Enum):
    linear = auto()
    polynomial = auto()
    gauss = auto()


def calculate_kernels(xs, kernel: Kernel, param=1):
    ks = []
    for i in range(len(xs)):
        ks.append([])
        for j in range(len(xs)):
            ks[i].append(calc_kernel(xs[i], xs[j], kernel, param))
    return ks


def calc_kernel(x1, x2, kernel: Kernel, param=1):
    result = {
        Kernel.linear: lambda a, b, p: calc_linear_kernel(a, b),
        Kernel.polynomial: lambda a, b, p: calc_poly_kernel(a, b, p),
        Kernel.gauss: lambda a, b, p: calc_gauss_kernel(a, b, p)
    }[kernel](x1, x2, param)
    return result


def calc_linear_kernel(x1, x2):
    return scalar_mult(x1, x2)


def calc_poly_kernel(x1, x2, p):
    return scalar_mult(x1, x2) ** p


def calc_gauss_kernel(x1, x2, p):
    diff = vector_diff(x1, x2)
    return math.exp(- p * scalar_mult(diff, diff))


def vector_diff(x1, x2):
    assert (len(x1) == len(x2))
    res = []
    for x_1, x_2 in list(zip(x1, x2)):
        res.append(x_1 - x_2)
    return res


def scalar_mult(x1, x2):
    assert (len(x1) == len(x2))
    res = 0
    for x_1, x_2 in list(zip(x1, x2)):
        res += x_1 * x_2
    return res

import smo
from kernels import Kernel
import numpy as np
import pandas as pd
import math
import random
import itertools
from tqdm import tqdm
from enum import Enum, auto
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def kernel_params(kernel):
    result = {
        Kernel.linear: [1],
        Kernel.polynomial: [2, 3, 4, 5],
        Kernel.gauss: [1, 2, 3, 4, 5]
    }[kernel]
    return result

def c_param():
    return [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

def all_kernels():
    return [Kernel.linear, Kernel.polynomial, Kernel.gauss]

def all_params_combinations(kernel: Kernel):
    return itertools.product(kernel_params(kernel), c_param())

chips_dataset = pd.read_csv('chips.csv')
geyser_dataset = pd.read_csv('geyser.csv')

chips_dataset
geyser_dataset

def class_to_int(ch):
    if (ch =='P'):
        return 1
    else:
        return -1

def dataset_to_local(dataset):
    rows = dataset.values.tolist()
    random.shuffle(rows)
    xs, ys = [], []
    for row in rows:
        clazz = row.pop()
        y = class_to_int(clazz)
        x = row
        xs.append(x)
        ys.append(y)
    return xs, ys


def split_in_chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def train_test(chunks, test_idx):
    train = []
    test = []
    for i in range(len(chunks)):
        if i != test_idx:
            train.extend(chunks[i])
        else:
            test.extend(chunks[i])
    return train, test

def train_test_blocks(xs, ys, k):
    xs_chunks = split_in_chunks(xs, k)
    ys_chunks = split_in_chunks(ys, k)
    xs_train_test = []
    ys_train_test = []
    assert(len(xs_chunks) == len(ys_chunks))
    assert(len(xs_chunks) == k)
    for i in range(k):
        xs_train_test.append(train_test(xs_chunks, i))
        ys_train_test.append(train_test(ys_chunks, i))
    return xs_train_test, ys_train_test

def handle_dataset_with_kernel(xs, ys, kernel: Kernel):
    params = all_params_combinations(kernel)
    best_model = smo.SVMModel(0, 0, [], [], Kernel.linear, 0)
    best_accuracy = 0
    for p, c in tqdm(params):
        xs_train_test, ys_train_test = train_test_blocks(xs, ys, 3)
        assert(len(xs_train_test) == len(ys_train_test))
        accuracy = 0
        for i in range(len(list(zip(xs_train_test, ys_train_test)))):
            xs_train, xs_test = xs_train_test[i]
            ys_train, ys_test = ys_train_test[i]
            model = smo.simplified_SMO(xs_train, ys_train, c, kernel, p)
            for x_test, y_test in zip(xs_test, ys_test):
                val = model.predict(x_test)
                if val == y_test:
                    accuracy += 1
        accuracy /= float(len(ys))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    return best_model, best_accuracy

xs_chips, ys_chips = dataset_to_local(chips_dataset)
xs_geyeser, ys_geyeser = dataset_to_local(geyser_dataset)


def draw(model, x_, y_, sc_x):
    x = np.array(x_)
    y = np.array(y_)
    sc_y = 0.01
    x_min, y_min = np.amin(x, 0)
    x_max, y_max = np.amax(x, 0)
    xs, ys = np.meshgrid(np.arange(x_min, x_max, sc_x), np.arange(y_min, y_max, sc_y))

    grid = np.c_[xs.ravel(), ys.ravel()]
    zs = np.apply_along_axis(lambda t: model.predict(t), 1, grid)
    zs = np.array(zs).reshape(xs.shape)

    plt.figure(figsize=(11, 11))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    x_neg, y_neg = x[y == -1].T
    x_pos, y_pos = x[y == 1].T

    plt.pcolormesh(xs, ys, zs, shading='nearest', cmap=ListedColormap(['#F54281', '#03A5FC']))
    plt.scatter(x_neg, y_neg, color='red', s=100)
    plt.scatter(x_pos, y_pos, color='blue', s=100)

    plt.show()
model_c_l, accuracy_c_l = handle_dataset_with_kernel(xs_chips, ys_chips, Kernel.linear)
print(f'Best accuracy: {accuracy_c_l}, c: {model_c_l.c}')
draw(model_c_l, xs_chips, ys_chips, 0.005)
model_c_p, accuracy_c_p = handle_dataset_with_kernel(xs_chips, ys_chips, Kernel.polynomial)
print(f'Best accuracy: {accuracy_c_p}, c: {model_c_p.c}, p: {model_c_p.param}')
draw(model_c_p, xs_chips, ys_chips, 0.005)
model_c_g, accuracy_c_g = handle_dataset_with_kernel(xs_chips, ys_chips, Kernel.gauss)
print(f'Best accuracy: {accuracy_c_g}, c: {model_c_g.c}, betta: {model_c_g.param}')
draw(model_c_g, xs_chips, ys_chips, 0.005)
model_g_p, accuracy_g_p = handle_dataset_with_kernel(xs_geyeser, ys_geyeser, Kernel.polynomial)
print(f'Best accuracy: {accuracy_g_p}, c: {model_g_p.c}, p: {model_g_p.param}')
draw(model_g_p, xs_geyeser, ys_geyeser, 1)
model_g_g, accuracy_g_g = handle_dataset_with_kernel(xs_geyeser, ys_geyeser, Kernel.gauss)
print(f'Best accuracy: {accuracy_g_g}, c: {model_g_g.c}, betta: {model_g_g.param}')
draw(model_g_g, xs_geyeser, ys_geyeser, 1)