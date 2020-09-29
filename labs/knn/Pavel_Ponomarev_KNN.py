from typing import List

def total_sum(cm):
    return sum(sum(cm, []))


def row_cols_sums(cm):
    row_sums = [0] * len(cm)
    col_sums = [0] * len(cm)
    for i in range(len(cm)):
        for j in range(len(cm)):
            row_sums[i] += cm[i][j]
            col_sums[j] += cm[i][j]
    return row_sums, col_sums


def calculate_f1(recall, precision):
    denominator = precision + recall
    numerator = 2 * precision * recall
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


def macro_micro_f(cm):
    row_sums, col_sums = row_cols_sums(cm)
    total = total_sum(cm)
    precision_total = 0
    recall_total = 0
    micro_f1 = 0
    for i in range(len(cm)):
        recall, precision = 0, 0
        weight = row_sums[i] / total
        if row_sums[i] > 0:
            recall = (cm[i][i] / row_sums[i]) * weight
        if col_sums[i] > 0:
            precision = (cm[i][i] / col_sums[i]) * weight
        precision_total = precision + precision_total
        recall_total = recall + recall_total
        micro_f1 = micro_f1 + calculate_f1(recall, precision)
    macro_f1 = calculate_f1(recall_total, precision_total)

    return macro_f1, micro_f1


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
                                   model: Model, max_class_num: int) -> int:
    classes_count_arr: List[float] = [0] * max_class_num
    for cl in range(max_class_num):
        modified_dataset = []
        for row in dataset:
            if row[1] == cl:
                modified_dataset.append((row[0], 1))
            else:
                modified_dataset.append((row[0], 0))

        classes_count_arr[cl] = non_parametric_regression(
            modified_dataset, target, model.distance, model.kernel, model.window, model.window_param)

    return classes_count_arr.index(max(classes_count_arr))


filename = 'ecoli.csv'
loaded_dataset = pd.read_csv(filename)


loaded_dataset


def minmax(dataset):
    result: List[Tuple[float, float]] = []
    for i in range(len(dataset[0]) - 1):
        result.append((dataset[:, i].min(), dataset[:, i].max()))
    return result


def normalize(dataset):
    new_dataset = dataset.copy()
    min_max = minmax(dataset)
    for row in new_dataset:
        for i in range(len(row) - 1):
            if min_max[i][1] - min_max[i][0] == 0:
                row[i] = row[i]
            else:
                row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
    return new_dataset


def classes_number(dataset: List[Tuple[List[float], int]]):
    classes = set()
    for row in dataset:
        classes.add(row[1])
    print(classes)
    return len(classes)


def shift_classes_if_needed(dataset):
    new_dataset = dataset.copy()
    cl = 1
    new_classes = {}
    for row in new_dataset:
        if row[len(row) - 1] not in new_classes:
            new_classes[row[len(row) - 1]] = cl
            cl = cl + 1
        row[len(row) - 1] = new_classes[row[len(row) - 1]]
    return new_dataset



norm_dataset = shift_classes_if_needed(normalize(loaded_dataset.values))


pd.DataFrame(norm_dataset)


max_dist = 0
for row1 in norm_dataset:
    for row2 in norm_dataset:
        max_dist = max(h.calculate_distance(row1[: len(row1) - 1], row2[:len(row2) - 1], h.Distance.euclidean),
                       max_dist)
max_dist


def all_models():
    models = []
    for window in h.all_windows():
        for kernel in h.all_kernels():
            for distance in h.all_distances():
                params = []
                if window == h.Window.variable:
                    params = range(1, 50, 8)
                else:
                    params = np.linspace(0.1, 1.5, 10)
                for param in params:
                    model = h.Model(distance, kernel, window, param)
                    models.append(model)
    return models



models = all_models()



def build_cm_naive(dataset: List[Tuple[List[float], int]], model: h.Model):
    n = classes_number(dataset)
    cm = [[0 for _ in range(n)] for _i in range(n)]
    for target_i in range(len(dataset)):
        obj, label = dataset[target_i]
        train_set = dataset[:target_i]
        if target_i != len(dataset) - 1:
            train_set += dataset[target_i + 1:]
        prediction = h.predict_class_of_target_naive(train_set, obj, model)
        cm[int(label) - 1][prediction - 1] += 1
    return cm


def build_cm_onehot(dataset: List[Tuple[List[float], int]], model: h.Model):
    n = classes_number(dataset)
    cm = [[0 for _ in range(n)] for _i in range(n)]
    for target_i in range(len(dataset)):
        obj, label = dataset[target_i]
        train_set = dataset[:target_i]
        if target_i != len(dataset) - 1:
            train_set += dataset[target_i + 1:]
        prediction = h.predict_class_of_target_onehot(train_set, obj, model, n)
        cm[int(label) - 1][prediction - 1] += 1
    return cm


def dataset_to_local_representation(dataset):
    values = dataset.tolist()
    my_dataset = []
    for row in values:
        attrs_row = []
        for i in (range(len(row) - 1)):
            attrs_row.append(row[i])
        label = row[len(row) - 1]
        my_dataset.append((attrs_row, label))
    return my_dataset


def classes_number(dataset: List[Tuple[List[float], int]]):
    classes = set()
    for row in dataset:
        classes.add(row[1])
    return len(classes)



local_dataset = dataset_to_local_representation(norm_dataset)
max_micro_f1 = 0
best_model_by_micro_f1 = models[0]
for model in tqdm(models):
    cm = build_cm_naive(local_dataset, model)
    _, micro_f1 = f.macro_micro_f(cm)
    if micro_f1 > max_micro_f1:
        best_model_by_micro_f1 = model
        max_micro_f1 = micro_f1

print(max_micro_f1)
print(best_model_by_micro_f1)
best_naive_model = best_model_by_micro_f1


cm = build_cm_naive(local_dataset, best_naive_model)
pd.DataFrame(cm, columns=[1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])


best_model_by_micro_f1 = models[0]
max_micro_f1 = 0
for model in tqdm(models):
    cm = build_cm_onehot(local_dataset, model)
    _, micro_f1 = f.macro_micro_f(cm)
    if micro_f1 > max_micro_f1:
        best_model_by_micro_f1 = model
        max_micro_f1 = micro_f1

print(max_micro_f1)
print(best_model_by_micro_f1)
best_onehot_model = best_model_by_micro_f1


cm = build_cm_onehot(local_dataset, best_onehot_model)
pd.DataFrame(cm, columns=[1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])


params_fixed = np.linspace(0.01, 1.5, 100)
f_micro_scores = []
for param in tqdm(params_fixed):
    cm = build_cm_naive(local_dataset, h.Model(h.Distance.euclidean, h.Kernel.tricube, h.Window.fixed, param))
    _, micro_f1 = f.macro_micro_f(cm)
    f_micro_scores.append(micro_f1)



plt.plot(params_fixed, f_micro_scores)



params_var: List[int] = range(1, 100)
f_micro_scores = []
for param in tqdm(params_fixed):
    cm = build_cm_onehot(local_dataset, h.Model(h.Distance.manhattan, h.Kernel.uniform, h.Window.variable, int(param)))
    _, micro_f1 = f.macro_micro_f(cm)
    f_micro_scores.append(micro_f1)


plt.plot(params_fixed, f_micro_scores)
