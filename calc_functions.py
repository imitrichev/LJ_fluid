from math import exp
from dataclasses import dataclass
from numpy import ones, array
from scipy.optimize import minimize


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

    def __init__(self, temps, visc_pref, x0):
        self.temps = temps
        self.visc_pref = visc_pref
        self.params = [0, 0, 0, 0, 0, 0, 0]
        self.x0 = x0
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


class Density:
    def __init__(self, rho0, temp0, p0):
        self.rho0 = rho0
        self.temp0 = temp0
        self.p0 = p0
        self.beta = 0
        self.e = 0

    def calc_rho1(self, t):
        return self.rho0 / (1 + self.beta * (t - self.temp0))

    def aad_rho1(self, beta, temp, rho):
        rho_calc = [self.rho0 / (1 + beta * (i - self.temp0)) for i in temp]
        return aad(rho, rho_calc)

    def setup_rho1(self, temps, rho):
        args = (temps, rho)
        test = minimize(fun=self.aad_rho1, x0=[0], args=args)
        self.beta = test.x[0]


def aad(x_exp, x_calc):
    assert (len(x_exp) == len(x_calc))
    returnable = 0
    for i in range(len(x_exp)):
        returnable += abs((x_exp[i] - x_calc[i]) / x_exp[i])
    returnable *= 100 / len(x_exp)
    return returnable


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


def test(x):
    returnable = 0
    for i in range(len(x)):
        returnable += x[i] ** 2 - i + 1
    return returnable


if __name__ == "__main__":
    a = VFT()
    print(a.calcVisc(60, 0.129))
