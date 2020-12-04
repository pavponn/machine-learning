

k = int(input())
n = int(input())
objects = []
for i in range(n):
    x_i, y_i = [int(a) for a in input().split()]
    objects.append((x_i, y_i))

left, right = 0, 0
for obj in objects:
    left += (obj[1] ** 2) / n
p_x = [0 for _ in range(k + 1)]
e_y_by_x = [0 for _ in range(k + 1)]
for obj in objects:
    p_x[obj[0]] += 1.0 / (n + 0.0)
    e_y_by_x[obj[0]] += obj[1] / (n + 0.0)

for i in range(k + 1):
    if p_x[i] == 0:
        continue
    right += e_y_by_x[i] * e_y_by_x[i] / p_x[i]

print(left - right)


