from math import exp
from dataclasses import dataclass
from numpy import ones
from scipy.optimize import least_squares


@dataclass
class ParamsDEF:
    d: float
    e: float
    f: float


@dataclass
class ParamsG:
    g0: float
    g1: float
    g2: float


class VFT:
    pref = 0.128

    def setupDEF(self):
        self.D, self.E, self.F = least_squares(aadDEF, ones(3), method='lm').x

    def setupHG(self):
        self.G0, self.G1, self.G2, self.H = least_squares(aadHG, ones(4), args=([self.D, self.E, self.F, self.pref]),
                                                          method='lm').x

    def __init__(self):
        self.setupDEF()
        self.setupHG()

    def calcVisc(self, t, p):
        return self.D * ((p + self.G0 + self.G1 * t + self.G2 * t ** 2) / (
                    self.pref + self.G0 + self.G1 * t + self.G2 * t ** 2)) ** self.H * exp(self.E / (t - self.F))


def aad(x_exp, x_calc):
    assert (len(x_exp) == len(x_calc))
    returnable = 0
    for i in range(len(x_exp)):
        returnable += abs((x_exp[i] - x_calc[i]) / x_exp[i])
    returnable *= 100 / len(x_exp)
    return returnable


def DEF(d, e, f, temp):
    return d * exp(e / (f - temp))


def aadDEF(x):
    temp = [125, 100, 80, 60, 40, 0, -20]
    visc = [1.9, 3.231, 4.982, 9.158, 21.987, 413.875, 209.777]
    visccalc = [DEF(x[0], x[1], x[2], temp[i]) for i in range(len(temp))]
    return aad(visc, visccalc), 0, 0


# Замена наименований параметров в связи с требованиями scipy
def HG(x, y, z, h, d, e, f, t, p, pref):
    return d * ((p + x + y * t + z * t ** 2) / (pref + x + y * t + z * t ** 2)) ** h * exp(e / (t - f))


def aadHG(x, d, e, f, pref):
    temp = [125]
    p = [14.7]
    visc = [0.8368]
    visccalc = [HG(x[0], x[1], x[2], x[3], d, e, f, temp[i], p[i], pref) for i in range(len(temp))]
    return aad(visc, visccalc), 0, 0, 0


def test(x):
    returnable = 0
    for i in range(len(x)):
        returnable += x[i] ** 2 - i + 1
    return returnable


if __name__ == "__main__":
    a = VFT()
    print(a.calcVisc(60, 0.129))
