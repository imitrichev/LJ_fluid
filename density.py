from common import aad, load_from_directory, aad_new, ViscData
from calc_functions import rho_t, rho_p, rho_full
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import arange


class Density:
    def __init__(self, rho0: float, point0: list[float]):
        self.rho0 = rho0
        self.p0, self.t0 = point0
        self.beta = 0
        self.e = 3

    def calc_rho_t(self, t: float) -> float:
        return rho_t(self.beta, [self.rho0, self.t0], t)

    def calc_rho_p(self, p: float) -> float:
        return rho_p(self.beta, [self.rho0, self.p0], p)

    def calc_rho_full(self, point: list[float]) -> float:
        p, t = point
        return rho_full([self.e, self.beta], [self.rho0, self.p0, self.t0], [p, t])

    def setup_rho_t(self, temps: list[float], rho: list[float]) -> bool:
        args = (rho_t, temps, rho, [self.rho0, self.t0])
        result = minimize(fun=aad_new, x0=0, args=args)
        self.beta = result.x[0]
        return result.success

    def setup_rho_p(self, p: list[float], rho: list[float]) -> bool:
        args = (rho_p, p, rho, [self.rho0, self.p0])
        result = minimize(fun=aad_new, x0=3, args=args)
        self.e = result.x[0]
        return result.success

    def setup_rho_full(self, rho, points) -> bool:
        static_params = (self.rho0, self.p0, self.t0)
        args = (rho_full, points, rho, static_params)
        print(aad_new([3, 0], rho_full, points, rho, static_params))
        result = minimize(fun=aad_new, method='Nelder-Mead', x0=[3, 0], args=args,
                          options={'maxiter': 100000, 'maxfev': 100000})
        self.e, self.beta = result.x
        print(aad_new([self.e, self.beta], rho_full, points, rho, static_params))
        print(result.message)
        return result.success


if __name__ == "__main__":
    substance = "R22"
    substance_data = ViscData(substance)
    points_to_process = substance_data.get_all_points(0)
    rho = [i[2] for i in points_to_process]
    points = [i[:2] for i in points_to_process]
    rh0, p0, t0 = substance_data.get_middle_point(0)
    density_calc = Density(rh0, [p0, t0])
    if not density_calc.setup_rho_full(rho, points):
        print('Error while optimizing')

    fig, ax = plt.subplots(2, 4)
    p_arr = list(substance_data.density.keys())
    ax[0][0].scatter(x=p_arr, y=[substance_data.density[i][t0] for i in p_arr])
    calc_p = arange(0.04, 0.101, 0.001)
    calc_rho = [density_calc.calc_rho_full([i, t0]) for i in calc_p]
    ax[0][0].plot(calc_p, calc_rho, "r", label="Расчётные значения")
    for i in range(len(p_arr)):
        p_cur = p_arr[i]
        graphy = (i + 1) // 4
        graphx = (i + 1) % 4
        exp_values = substance_data.get_by_pressure(0, p_arr[i])
        exp_temps = [j[1] for j in exp_values]
        exp_rho = [j[2] for j in exp_values]
        step = 0.01
        t_calc = arange(exp_temps[0], exp_temps[-1] + step, step)
        rho_calc = [density_calc.calc_rho_full([p_cur, j]) for j in t_calc]
        ax[graphy][graphx].scatter(x=exp_temps, y=exp_rho, label="Экспериментальные данные")
        ax[graphy][graphx].plot(t_calc, rho_calc, "r", label="Расчётные значения")
        ax[graphy][graphx].set_title("p = %.2f MPa" % p_cur)
        ax[graphy][graphx].set_ylabel("Rho, м3/кг")
        ax[graphy][graphx].set_xlabel("P, МПа")
        ax[graphy][graphx].legend()

    print("Beta = %f" % density_calc.beta)
    print("E = %f" % density_calc.e)
    plt.show()
