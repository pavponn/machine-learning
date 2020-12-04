
k1, k2 = [int(x) for x in input().split()]
n = int(input())

objects = []
for i in range(n):
    x_i, y_i = [int(a) for a in input().split()]
    objects.append((x_i, y_i))
ex1, ex2 = [0 for _ in range(k1 + 1)], [0 for _ in range(k2 + 1)]
dic = {}
for i in range(n):
    ex1[objects[i][0]] = ex1[objects[i][0]] + 1
    ex2[objects[i][1]] = ex2[objects[i][1]] + 1
    dic[objects[i]] = dic.get(objects[i], 0) + 1

for i in range(len(ex1)):
    ex1[i] *= 1 / n

for i in range(len(ex2)):
    ex2[i] *= 1 / n

res = n

for item in dic.items():
    m = n * ex1[item[0][0]] * ex2[item[0][1]]
    p = ((item[1] - m) ** 2) / m
    res = res - m + p

print(res)
