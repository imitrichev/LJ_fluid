import math


def aad(x_exp, x_calc):
    assert (len(x_exp) == len(x_calc))
    returnable = 0
    for i in range(len(x_exp)):
        returnable += abs((x_exp[i] - x_calc[i]) / x_exp[i])
    returnable *= 100 / len(x_exp)
    return returnable


def linesum(G, T, length):
    returnable = 0
    for i in range(length):
        returnable += G[i] * T ** i
    return returnable


# Найти и вставить все параметры
def vft(p, T):
    D = pref = E = F = H = 0
    G = [0, 0, 0]
    return D * (p + linesum(G, T, 3)) / (pref + linesum(G, T, 3)) ** H * math.exp(E / (T - F))


def test(x):
    returnable = 0
    for i in range(len(x)):
        returnable += x[i] ** 2 - i + 1
    return returnable
