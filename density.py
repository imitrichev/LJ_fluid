from common import aad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from os import listdir
from numpy import arange


class Density:
    def __init__(self, rho0, t0, p0):
        self.rho0 = rho0
        self.t0 = t0
        self.p0 = p0
        self.beta = 0
        self.e = 10

    def calc_rho1(self, t, beta=None):
        if not beta:
            beta = self.beta
        return self.rho0 / (1 + beta * (t - self.t0))

    def calc_rho2(self, p, e=None):
        if not e:
            e = self.e
        return self.rho0 / (1 - (p - self.p0) / e)

    def calc_rho_full(self, p, t):
        return (self.rho0 / (1 + self.beta * (t - self.t0))) / (1 - (p - self.p0) / self.e)

    def aad_rho1(self, beta, temp, rho):
        rho_calc = [self.calc_rho1(i, beta) for i in temp]
        return aad(rho, rho_calc)

    def aad_rho2(self, e, p, rho):
        rho_calc = [self.calc_rho2(i, e) for i in p]
        test = aad(rho, rho_calc)
        return test

    def setup_rho1(self, temps, rho):
        args = (temps, rho)
        test = minimize(fun=self.aad_rho1, x0=0, args=args)
        self.beta = test.x[0]

    def setup_rho2(self, p, rho):
        args = (p, rho)
        test = minimize(fun=self.aad_rho2, x0=3, args=args)
        self.e = test.x[0]


if __name__ == "__main__":
    rho = {}
    substance = "R22"
    for file in listdir(substance):
        with open(substance + "\\" + file, "r") as f:
            text = f.read()
            p = float(file[:-4])
            rho[p] = {}
            for line in text.split('\n'):
                cur_line = line.split(' ')
                rho[p][float(cur_line[0])] = float(cur_line[2])
    p = list(rho.keys())
    rho_p_dynamic = [rho[i][80] for i in p]
    graph_numb = [0, 0]
    p0 = 1
    temps = list(rho[p0].keys())
    t0 = 80
    rho_t = list(rho[p0].values())
    rho_p = [rho[i][t0] for i in p]
    rho0 = rho[p0][t0]
    test = Density(rho0, t0, p0)
    test.setup_rho1(temps, rho_t)
    test.setup_rho2(p, rho_p)
    print(test.e)
    print(test.aad_rho2(test.e, p, rho_p))
    fig, ax = plt.subplots(2, 4)
    ax[0][0].plot(arange(0.3, 3.1, 0.1), [test.calc_rho2(i) for i in arange(0.3, 3.1, 0.1)], "r", label="Calculated")
    ax[0][0].plot(p, [rho[i][80] for i in p], 'b', label="Expected")
    ax[0][0].set_title("Rho(p)")
    for i in range(len(p)):
        p_cur = p[i]
        graphy = (i + 1) // 4
        graphx = (i + 1) % 4
        exp_temps = list(rho[p_cur].keys())
        t_arr = arange(exp_temps[0], exp_temps[-1], 0.5)
        ax[graphy][graphx].plot(t_arr, [test.calc_rho_full(p_cur, j) for j in t_arr], "r", label="Calculated")
        ax[graphy][graphx].plot(exp_temps, list(rho[p_cur].values()), "b", label="Expected")
        ax[graphy][graphx].set_title("P = %f" % p_cur)
    print("Beta = %f" % test.beta)
    print("E = %f" % test.e)
    plt.legend()
    plt.show()
