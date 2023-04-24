from math import exp
from dataclasses import dataclass
from common import aad, ViscData, aad_new
from scipy.optimize import minimize, Bounds
from os import listdir
import matplotlib.pyplot as plt
from numpy import arange


class VFT:
    def __init__(self, mu_data: dict[float:dict[float:float]], pref: float):
        self.mu_data = mu_data
        self.pref = pref
        self.params = [0, 0, 0, 0, 0, 0, 0]

    def setup_def(self):
        visc_pref = list(self.mu_data[pref].values())
        temps = list(self.mu_data[pref].keys())

        temp_x = [1e-3, 1000, 665.2620524970494, 0, 0, 0, 0]
        if visc_pref[0] < visc_pref[-1]:
            temp_x = [1e-3, 8000, 665.2620524970494, 0, 0, 0, 0]

        x0 = temp_x
        args = (DEF, temps, visc_pref)
        test = minimize(fun=aad_new, x0=x0[:3], method='Nelder-Mead', args=args, tol=1e-7)
        self.params[0], self.params[1], self.params[2] = test.x

    def setup_hg(self):
        self.params[3], self.params[4], self.params[5], self.params[6] = minimize(aadHG, self.x0[3:], args=(
            [self.params[0], self.params[1], self.params[2], self.pref])).x

    def calc_visc_def(self, t):
        return DEF(self.params[:3], t)

    def calcVisc(self, t, p):
        return HG(self.params[3], self.params[4], self.params[5], self.params[6], self.params[0], self.params[1],
                  self.params[2], t, p, self.pref)


class MuMix:
    def __init__(self, ):
        pass


def DEF(params, point):
    d, e, f = params
    return d * exp(e / (point + 273.15 - f))


# Замена наименований параметров в связи с требованиями scipy
def HG(params, static_params, point):
    g0, g1, g2, h = params
    d, e, f, pref = static_params
    t, p = point
    return d * ((p + g0 + g1 * t + g2 * t ** 2) / (pref + g0 + g1 * t + g2 * t ** 2)) ** h * exp(e / (t - f))


def aadHG(params, static_params, points, visc):
    visccalc = [HG(params, static_params, points[i]) for i in range(len(points))]
    return aad(visc, visccalc), 0, 0, 0


if __name__ == "__main__":
    substance = "R115"
    data = ViscData(substance)
    pref = 0.06

    visc = data.visc_k
    temps = list(visc[pref].keys())
    viscs = [visc[pref][i] for i in temps]

    test = VFT(visc, pref)
    test.setup_def()
    print(f'D = {test.params[0]}\nE = {test.params[1]}\nF = {test.params[2]}\n')
    print(f'AAD = {aadDEF(test.params[:3], temps, viscs)}')

    fig, ax = plt.subplots()
    ax.scatter(temps, viscs, label="Экспериментальные данные")
    ax.plot(arange(temps[0], temps[-1] + 1), [test.calc_visc_def(i) for i in arange(temps[0], temps[-1] + 1)], "r",
            label="Расчётные значения")
    ax.set_ylabel("Mu, cP")
    ax.set_xlabel("T, Гр. Цельсия")
    ax.set_title("mu = f(t)")
    plt.grid()
    plt.legend()
    plt.show()
