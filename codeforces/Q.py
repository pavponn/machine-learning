import math

res = 0
k1, k2 = [int(x) for x in input().split()]
n = int(input())

objects = []
for i in range(n):
    x_i, y_i = [int(a) for a in input().split()]
    objects.append((x_i, y_i))

prob_x = {}
prob_xy = {}


for i in range(n):
    prob_x[objects[i][0]] = prob_x.get(objects[i][0], 0) + 1. / n
    prob_xy[objects[i]] = prob_xy.get(objects[i], 0) + 1. / n

for item in prob_xy.items():
    res = res - item[1] * (math.log(item[1]) - math.log(prob_x[item[0][0]]))

print(res)
