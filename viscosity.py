from math import exp
from dataclasses import dataclass
from common import aad
from scipy.optimize import minimize
from os import listdir
import matplotlib.pyplot as plt
from numpy import arange


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

    def __init__(self, temps, visc_pref, x0, pref=0):
        self.temps = temps
        self.visc_pref = visc_pref
        self.params = [0, 0, 0, 0, 0, 0, 0]
        self.x0 = x0
        self.pref = pref
        self.setup_def()

    def setup_def(self):
        args = (self.temps, self.visc_pref)
        test = minimize(fun=aadDEF, x0=self.x0[:3], args=args)
        self.params[0], self.params[1], self.params[2] = test.x

    def setup_hg(self):
        self.params[3], self.params[4], self.params[5], self.params[6] = minimize(aadHG, self.x0[3:], args=(
            [self.params[0], self.params[1], self.params[2], self.pref]), method='lm').x

    def calc_visc_def(self, t):
        return DEF(self.params[0], self.params[1], self.params[2], t)

    def calcVisc(self, t, p):
        return HG(self.params[3], self.params[4], self.params[5], self.params[6], self.params[0], self.params[1],
                  self.params[2], t, p, self.pref)


def DEF(d, e, f, t):
    return d * exp(e / (f - (t + 273.15)))


def aadDEF(x, temp, visc):
    visccalc = [DEF(x[0], x[1], x[2], temp[i]) for i in range(len(temp))]
    aad_calc = aad(visc, visccalc)
    return aad_calc


# Замена наименований параметров в связи с требованиями scipy
def HG(x, y, z, h, d, e, f, t, p, pref):
    return d * ((p + x + y * t + z * t ** 2) / (pref + x + y * t + z * t ** 2)) ** h * exp(e / (t - f))


def aadHG(x, d, e, f, pref):
    temp = [125]
    p = [14.7]
    visc = [0.8368]
    visccalc = [HG(x[0], x[1], x[2], x[3], d, e, f, temp[i], p[i], pref) for i in range(len(temp))]
    return aad(visc, visccalc), 0, 0, 0


if __name__ == "__main__":
    visc = {}
    substance = "R22"
    for file in listdir(substance):
        with open(substance + "\\" + file, "r") as f:
            text = f.read()
            p = float(file[:-4])
            visc[p] = {}
            for line in text.split('\n'):
                cur_line = line.split(' ')
                visc[p][float(cur_line[0])] = float(cur_line[-1])

    x0 = [10, 1000, 100, 0, 0, 0, 0]
    p0 = 0.8
    temps = list(visc[p0].keys())
    visc = list(visc[p0].values())
    test = VFT(temps, visc, x0)
    print(test.params[0])
    print(test.params[1])
    print(test.params[2])

    t_np = arange(temps[0], temps[-1], 0.5)

    fig, ax = plt.subplots()
    ax.plot(temps, visc, "b", label="Expected")
    ax.plot(t_np, [test.calc_visc_def(i) for i in t_np], "r", label="Calculated")
    plt.legend()
    plt.show()
