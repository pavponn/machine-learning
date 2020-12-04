from numpy import mean
import math

def cov(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    s = 0
    for i in range(len(x)):
        s += (x[i] - mean_x) * (y[i] - mean_y)

    return (1. / len(x)) * s


def disp(x):
    mean_x = mean(x)
    s = 0
    for i in range(len(x)):
        s += (x[i] - mean_x) ** 2

    return (1. / len(x)) * s


def main():
    n = int(input())
    x, y = [], []
    for _ in range(n):
        line = [int(x) for x in input().split()]
        x.append(line[0])
        y.append(line[1])

    dx = disp(x)
    dy = disp(y)
    if math.fabs(dx) < 1e-7 or math.fabs(dy) < 1e-7:
        print(0)
        return
    print(cov(x, y) / math.sqrt(dx * dy))


main()
