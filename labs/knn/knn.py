from typing import Tuple, List

import pandas as pd
import numpy as np

import helper as h


def classes_number(dataset: List[Tuple[List[float], int]]):
    classes = set()
    for row in dataset:
        classes.add(row[1])
    return len(classes)


def build_cm_naive(dataset: List[Tuple[List[float], int]], model: h.Model):
    n = classes_number(dataset)
    cm = [[0 for _ in range(n)] for _i in range(n)]
    for target_i in range(len(dataset)):
        obj, label = dataset[target_i]
        train_set = dataset[:target_i] + dataset[target_i:]
        prediction = h.predict_class_of_target_naive(dataset, obj, model)
        cm[label][prediction] += 1
    return cm


filename = 'ecoli.csv'
loaded_dataset = pd.read_csv(filename)

