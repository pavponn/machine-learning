from enum import Enum, auto
import math as math
from typing import Tuple, List


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


class Model(object):

    def __init__(self, distance: Distance, kernel: Kernel, window: Window, window_param):
        self.distance = distance
        self.kernel = kernel
        self.window = window
        self.window_param = window_param

    def __str__(self):
        return f'Model: {self.distance} {self.kernel} {self.window} {self.window_param}'

    def __repr__(self):
        return str(self)


def all_distances():
    return [Distance.manhattan, Distance.euclidean, Distance.chebyshev]


def all_kernels():
    return [Kernel.uniform, Kernel.triangular,
            Kernel.epanechnikov, Kernel.quartic,
            Kernel.triweight, Kernel.tricube,
            Kernel.gaussian, Kernel.cosine,
            Kernel.logistic, Kernel.sigmoid]


def all_windows():
    return [Window.fixed, Window.variable]


def all_models():
    models = []
    for window in all_windows():
        for kernel in all_kernels():
            for distance in all_distances():
                params = []
                if window == Window.variable:
                    params = range(1, 101, 10)
                else:
                    params = np.linspace(0.1, 4, 10)
                for param in params:
                    model = Model(distance, kernel, window, param)
                    models.append(model)
    return models


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



# dataset = ([...attrs], label)
# target = [...attrs]
# distance
def non_parametric_regression(dataset: List[Tuple[List[float], int]], target,
                              distance: Distance, kernel: Kernel,
                              window: Window, window_param):
    sorted_data = \
        sorted(
            dataset,
            key=lambda x: calculate_distance(x[0], target, distance)
        )
    result = 0
    if window == Window.variable:
        window_param = calculate_distance(sorted_data[window_param][0], target, distance)
    if window_param != 0:
        up = 0
        down = 0
        for row in sorted_data:
            u = calculate_distance(row[0], target, distance) / window_param
            k = calculate_kernel_value(u, kernel)
            up += row[1] * k
            down += k
        if down == 0:
            result = sum([row[1] for row in sorted_data]) / len(sorted_data)
        else:
            result = up / down
    else:
        if sorted_data[0][0] == target:
            same = []
            for d in sorted_data:
                if target == d[0]:
                    same.append(d[1])
            result = sum(same) / len(same)
        else:
            result = sum([row[1] for row in dataset]) / len(dataset)
    return result


def predict_class_of_target_naive(dataset: List[Tuple[List[float], int]], target: List[float],
                                  model: Model) -> int:
    return round(
        non_parametric_regression(dataset, target, model.distance, model.kernel, model.window, model.window_param)
    )


def predict_class_of_target_onehot(dataset: List[Tuple[List[float], int]], target: List[float],
                                   model, max_class_num) -> int:
    sorted_data = \
        sorted(
            dataset,
            key=lambda x: calculate_distance(x[0], target, model.distance)
        )
    if model.window_param == Window.variable:
        window_param = calculate_distance(sorted_data[model.window_param][0], target, model.distance)
    else:
        window_param = model.window_param
    classes_count_arr = [0] * max_class_num  # TODO: class num
    for row in dataset:
        u = calculate_distance(row[0], target, model.distance) / window_param
        real_class = row[1]
        classes_count_arr[int(real_class)] += calculate_kernel_value(u, model.kernel)
    return classes_count_arr.index(max(classes_count_arr))
