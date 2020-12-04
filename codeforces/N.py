from typing import Tuple, List


def cmp_to_key(mycmp):
    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return K


def comparator1(f, s):
    if f[0] - s[0] == 0:
        return f[1] - s[1]
    else:
        return f[0] - s[0]


def comparator2(f, s):
    if f[1] - s[1] == 0:
        return f[0] - s[0]
    else:
        return f[1] - s[1]


def inter_class_dist(objects):
    n = len(objects)
    suf_sum = 0
    pref_sum = 0
    res = 0
    sorted_objects = sorted(objects, key=cmp_to_key(comparator1))
    pref = {}
    suf = {}
    for i in range(n):
        pref[sorted_objects[i][1]] = (0, 0)
        cur = suf.get(sorted_objects[i][1], (0, 0))
        suf[sorted_objects[i][1]] = (cur[0] + sorted_objects[i][0], cur[1] + 1)
        suf_sum += sorted_objects[i][0]
    for i in range(n):
        suf_sum -= sorted_objects[i][0]
        suf[sorted_objects[i][1]] = (
        suf[sorted_objects[i][1]][0] - sorted_objects[i][0], suf[sorted_objects[i][1]][1] - 1)
        res += (suf_sum - suf[sorted_objects[i][1]][0]) - (n - (i + 1) - suf[sorted_objects[i][1]][1]) * \
               sorted_objects[i][0] + \
               (i - pref[sorted_objects[i][1]][1]) * sorted_objects[i][0] - (pref_sum - pref[sorted_objects[i][1]][0])

        pref[sorted_objects[i][1]] = (
        pref[sorted_objects[i][1]][0] + sorted_objects[i][0], pref[sorted_objects[i][1]][1] + 1)
        pref_sum += sorted_objects[i][0]
    return res


def intra_class_dist(objects: List[Tuple[int, int]], k):
    res = 0
    n = len(objects)
    sorted_objects = sorted(objects, key=cmp_to_key(comparator2))
    i, j = 0, 0
    for cl in range(1, k + 1):
        left, right = 0, 0
        while True:
            if j >= n or sorted_objects[j][1] != cl:
                break
            right += sorted_objects[j][0]
            j = j + 1

        for it in range(i, j):
            right = right - sorted_objects[it][0]
            res += right - left + (it - i) * sorted_objects[it][0] - (j - 1 - it) * sorted_objects[it][0]
            left = left + sorted_objects[it][0]
        i = j

    return res


def main():
    k = int(input())
    n = int(input())
    objects = []
    for i in range(n):
        x_i, y_i = [int(a) for a in input().split()]
        objects.append((x_i, y_i))
    print(intra_class_dist(objects, k))
    print(inter_class_dist(objects))


main()
