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

