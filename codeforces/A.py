n, m, k = [int(x) for x in input().split()]
classes = [int(x) for x in input().split()]

classes_with_indexes = list(zip(classes, range(1, n + 1)))

sorted_classes_with_indexes = sorted(classes_with_indexes, key=lambda x: x[0])

result = [[] for _ in range(k)]

for i in range(n):
    result[i % k].append(sorted_classes_with_indexes[i][1])

for block in result:
    print(len(block), *sorted(block))
