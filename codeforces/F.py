from typing import Dict, Tuple, Set
import math

EPS = 1e-10


def calculate_likelihood_probabilities(k: int, alpha: int, classes_num: Dict[int, int], all_words: Set[str],
                                       word_in_class_k: Dict[Tuple[str, int], int]) -> Dict[Tuple[str, int], Tuple[float, float]]:

    likelihood_probabilities: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for word in all_words:
        for cl in range(k):
            count = 0
            if (word, cl) in word_in_class_k:
                count = word_in_class_k[(word, cl)]
            numerator = count + alpha + 0.000
            denominator = classes_num[cl] + alpha * 2 + 0.0000
            likelihood_probabilities[(word, cl)] = (numerator, denominator)
    return likelihood_probabilities


def solve():
    k = int(input())
    lambdas = [int(x) for x in input().split()]
    alpha = int(input())
    n = int(input())
    all_words: Set[str] = set()
    word_in_class_k: Dict[Tuple[str, int], int] = {}
    classes_num = {}
    for cl in range(k):
        classes_num[cl] = 0

    for _ in range(n):
        line = [x for x in input().split()]
        this_class = int(line.pop(0)) - 1
        this_length = int(line.pop(0))
        this_words = set(line)

        for w in this_words:
            all_words.add(w)
        classes_num[this_class] += 1
        for word in this_words:
            if (word, this_class) not in word_in_class_k:
                word_in_class_k[(word, this_class)] = 0
            word_in_class_k[(word, this_class)] += 1

    likelihood_probabilities: Dict[Tuple[str, int], Tuple[float, float]] = \
        calculate_likelihood_probabilities(k, alpha, classes_num, all_words, word_in_class_k)

    m = int(input())
    for i in range(m):
        line = [x for x in input().split()]
        this_length = int(line.pop(0))
        this_words = set(line)
        num = [0] * k
        for cl in range(k):
            this_num = 0
            this_num += math.log(lambdas[cl] * (EPS + classes_num[cl] / n))
            for word in all_words.difference(this_words):
                this_num += math.log(1 - (likelihood_probabilities[(word, cl)][0] / likelihood_probabilities[(word, cl)][1]))
            for word in this_words.intersection(all_words):
                this_num += math.log(likelihood_probabilities[(word, cl)][0] / likelihood_probabilities[(word, cl)][1])
            num[cl] = this_num
        max_num = max(num)
        snd = 0
        for ln_pr in num:
            snd += math.exp(ln_pr - max_num)
        snd = math.log(snd) + max_num
        for cl in range(k):
            if cl != k - 1:
                print(math.exp(num[cl] - snd), end=' ')
            else:
                print(math.exp(num[cl] - snd))


solve()