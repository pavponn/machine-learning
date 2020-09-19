from enum import Enum, auto
import math as math


class Distance(Enum):
    manhattan = auto()
    euclidean = auto()
    chebyshev = auto()


class Kernel(Enum):
    uniform = auto()
    triangular = auto()
    epanechnikov = auto()
    quartic = auto()
    triweight = auto()
    tricube = auto()
    gaussian = auto()
    cosine = auto()
    logistic = auto()
    sigmoid = auto()


class Window(Enum):
    fixed = auto()
    variable = auto()


def calc_dist_manhattan(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist = dist + abs(x1[i] - x2[i])
    return dist


def calc_dist_euclidean(x1, x2):
    return math.dist(x1, x2)


def calc_dist_chebyshev(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist = max(abs(x1[i] - x2[i]), dist)
    return dist


def calculate_distance(x1, x2, distance):
    result = {
        Distance.manhattan: lambda a, b: calc_dist_manhattan(a, b),
        Distance.euclidean: lambda a, b: calc_dist_euclidean(a, b),
        Distance.chebyshev: lambda a, b: calc_dist_chebyshev(a, b)
    }[distance](x1, x2)
    return result


def calc_kernel_uniform(u):
    if abs(u) < 1:
        return 0.5
    else:
        return 0


def calc_kernel_triangular(u):
    if abs(u) < 1:
        return 1. - abs(u)
    else:
        return 0


def calc_kernel_epanechnikov(u):
    if abs(u) < 1:
        return (3. / 4) * (1. - (u ** 2))
    else:
        return 0


def calc_kernel_quartic(u):
    if abs(u) < 1:
        return (15. / 16) * ((1. - (u ** 2)) ** 2)
    else:
        return 0


def calc_kernel_triweight(u):
    if abs(u) < 1:
        return (35. / 32) * ((1. - (u ** 2)) ** 3)
    else:
        return 0


def calc_kernel_tricube(u):
    if abs(u) < 1:
        return (70. / 81) * ((1. - (abs(u) ** 3)) ** 3)
    else:
        return 0


def calc_kernel_gaussian(u):
    return (1. / (math.sqrt(2 * math.pi))) * math.exp((-0.5) * (u ** 2))


def calc_kernel_cosine(u):
    if abs(u) < 1:
        return (math.pi * math.cos(math.pi * u / 2)) / 4
    else:
        return 0


def calc_kernel_logistic(u):
    return 1. / (math.exp(u) + 2. + math.exp(-u))


def calc_kernel_sigmoid(u):
    return (2. / math.pi) / (math.exp(u) + math.exp(-u))


def calculate_kernel_value(u, kernel):
    result = {
        Kernel.uniform: lambda _u: calc_kernel_uniform(_u),
        Kernel.triangular: lambda _u: calc_kernel_triangular(_u),
        Kernel.epanechnikov: lambda _u: calc_kernel_epanechnikov(_u),
        Kernel.quartic: lambda _u: calc_kernel_quartic(_u),
        Kernel.triweight: lambda _u: calc_kernel_triweight(_u),
        Kernel.tricube: lambda _u: calc_kernel_tricube(_u),
        Kernel.gaussian: lambda _u: calc_kernel_gaussian(_u),
        Kernel.cosine: lambda _u: calc_kernel_cosine(_u),
        Kernel.logistic: lambda _u: calc_kernel_logistic(_u),
        Kernel.sigmoid: lambda _u: calc_kernel_sigmoid(_u),
    }[kernel](u)
    return result


n, m = [int(x) for x in input().split()]

dataset = []
for i in range(n):
    attrs = [int(x) for x in input().split()]
    label = attrs.pop()
    dataset.append((attrs, label))

q = [int(x) for x in input().split()]
distance = input()
kernel = input()
window = input()
window_param = int(input())


sorted_data = \
    sorted(
        dataset,
        key=lambda x: calculate_distance(x[0], q, Distance[distance])
    )

result = 0
if Window[window] == Window.variable:
    window_param = calculate_distance(sorted_data[window_param][0], q, Distance[distance])
if window_param != 0:
    up = 0
    down = 0
    for row in sorted_data:
        u = calculate_distance(row[0], q, Distance[distance]) / window_param
        k = calculate_kernel_value(u, Kernel[kernel])
        up += row[1] * k
        down += k
    if down == 0:
        result = sum([row[1] for row in sorted_data]) / n
    else:
        result = up / down
else:
    if sorted_data[0][0] == q:
        same = []
        for d in sorted_data:
            if q == d[0]:
                same.append(d[1])
        result = sum(same) / len(same)
    else:
        result = sum([row[1] for row in dataset]) / n


print(result)
