from common import aad, load_from_directory, aad_new, ViscData
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import arange


class Density:
    def __init__(self, rho0: float, t0: float, p0: float):
        self.rho0 = rho0
        self.t0 = t0
        self.p0 = p0
        self.beta = 0
        self.e = 10

    def calc_rho1(self, params, t):
        return self.rho0 / (1 + params * (t - self.t0))

    def calc_rho2(self, params, p):
        return self.rho0 / (1 - (p - self.p0) / params)

    def calc_rho_full(self, params, point):
        e, beta = params
        p, t = point
        return (self.rho0 / (1 + beta * (t - self.t0))) / (1 - (p - self.p0) / e)

    def setup_rho1(self, temps: list[float], rho: list[float]):
        args = (self.calc_rho1, temps, rho)
        test = minimize(fun=aad_new, x0=0, args=args)
        self.beta = test.x[0]

    def setup_rho2(self, p: list[float], rho: list[float]):
        args = (self.calc_rho2, p, rho)
        test = minimize(fun=aad_new, x0=3, args=args)
        self.e = test.x[0]

    def setup_rho_full(self, rho, points):
        args = (self.calc_rho_full, rho, points)
        test = minimize(fun=aad_new, method='Nelder-Mead', x0=[3, 0], args=args)
        self.e = test.x[0]
        self.beta = test.x[1]


if __name__ == "__main__":
    substance = "R22"
    rho = ViscData(substance).density
    p = list(rho.keys())
    graph_numb = [0, 0]
    p0 = 0.05
    temps = list(rho[p0].keys())
    t0 = -80
    rho_p_dynamic = [rho[i][t0] for i in p]
    rho_t = list(rho[p0].values())
    rho_p = [rho[i][t0] for i in p]
    rho0 = rho[p0][t0]
    test = Density(rho0, t0, p0)
    test.setup_rho_full(rho)
    fig, ax = plt.subplots(2, 4)

    ax[0][0].plot(p, [test.calc_rho_full(i, t0) for i in p], "r", label="Расчётные значения")
    ax[0][0].scatter(x=p, y=[rho[i][t0] for i in p], label="Экспериментальные данные")
    ax[0][0].set_ylim([1515, 1520])
    ax[0][0].set_ylabel("Rho, м3/кг")
    ax[0][0].set_xlabel("T, Гр. Цельсия")
    ax[0][0].set_title("rho = f(p)")
    ax[0][0].legend()
    print(f"expected: {[rho[i][t0] for i in p]}")
    print(f"calculated: {[test.calc_rho_full(i, t0) for i in p]}")
    for i in range(len(p)):
        p_cur = p[i]
        graphy = (i + 1) // 4
        graphx = (i + 1) % 4
        exp_temps = list(rho[p_cur].keys())
        t_arr = arange(exp_temps[0], exp_temps[-1] + 0.5, 0.5)
        ax[graphy][graphx].plot(t_arr, [test.calc_rho_full(p_cur, j) for j in t_arr], "r", label="Расчётные значения")
        ax[graphy][graphx].scatter(x=exp_temps, y=list(rho[p_cur].values()), label="Экспериментальные данные")
        ax[graphy][graphx].set_title("p = %.2f MPa" % p_cur)
        ax[graphy][graphx].set_ylabel("Rho, м3/кг")
        ax[graphy][graphx].set_xlabel("P, МПа")
        ax[graphy][graphx].legend()
    print("Beta = %f" % test.beta)
    print("E = %f" % test.e)
    plt.show()
