
def ranks(x):
    indexed_x = []
    for i in range(len(x)):
        indexed_x.append((x[i], i))
    sorted_x = sorted(indexed_x)
    rnk = 0
    rnks = [0 for i in range(len(x))]
    rnks[sorted_x[0][1]] = 0
    for i in range(1, len(x)):
        if sorted_x[i - 1][0] != sorted_x[i][0]:
            rnk += 1
        rnks[sorted_x[i][1]] = rnk
    return rnks


def spirman(x, y):
    res = 0
    ranks_x, ranks_y = ranks(x), ranks(y)
    for xr, yr in list(zip(ranks_x, ranks_y)):
        res += (xr - yr) ** 2
    n = len(ranks(x))
    return 1.0 - 6.0 * res / ((n - 1.0) * (n + 1.0) * n)


def main():
    n = int(input())
    x, y = [], []
    for _ in range(n):
        line = [int(x) for x in input().split()]
        x.append(line[0])
        y.append(line[1])
    print(spirman(x, y))


main()