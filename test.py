from calc_functions import VFT, Density
import matplotlib.pyplot as plt
from os import listdir
from numpy import arange

'''x0 = [0, -1600, 100, 0, 0, 0, 0]
temps = [-10, 10, 30, 50, 70]
visc = [0.000285283401162772, 0.000222047273756758, 0.000171654930617118, 0.000128470773935355, 0.0000853609416455425]
test = VFT(temps, visc, x0)
print(test.params[0])
print(test.params[1])
print(test.params[2])'''
rho = {}
for file in listdir('R115'):
    with open("R115\\" + file, "r") as f:
        text = f.read()
        p = float(file[:-4])
        rho[p] = {}
        for line in text.split('\n'):
            cur_line = line.split(' ')
            rho[p][float(cur_line[0])] = float(cur_line[2])
p = list(rho.keys())
fig, ax = plt.subplots(2, 4)
graph_numb = [0, 0]
for p_cur in p:
    p0 = p_cur
    temps = list(rho[p0].keys())
    t0 = temps[int(len(temps) / 2)]
    rho0 = rho[p0][t0]
    test = Density(rho0, t0, p0)
    test.setup_rho1(temps[::2], list(rho[p0].values())[::2])
    t = arange(temps[0], temps[-1], 0.5)
    rho_calc = [test.calc_rho1(i) for i in t]
    graphy = graph_numb[0]
    graphx = graph_numb[1]
    ax[graphy][graphx].plot(t, rho_calc, "b")
    ax[graphy][graphx].plot(temps, list(rho[p0].values()), "r")
    ax[graphy][graphx].set_title("График плотности при P = %f\nПарметр beta = %f" % (p0, test.beta))
    graph_numb[1] += 1
    if graph_numb[1] > 3:
        graph_numb = [graph_numb[0] + 1, 0]
plt.show()
