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
