# Задача. Предсказывание состава жидкой фазы смеси.
# Входные параметры: Температура C, давление Бар (МПа/10) для каждой точки, вязкость cP, плотность г/мл и состав газа
# R-22/R-115 (48.8/51.2) ------> (0.488/0.512)
# Построить зависимость концентрации от давления

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import numpy as np
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, FlashVL
from thermo.interaction_parameters import IPDB
from scipy.optimize import minimize

class Oil:
    def __init__(self, M_oil, T_oil, P_oil, Omega_oil):
        self.M_oil = M_oil
        self.T_oil = T_oil
        self.P_oil = P_oil
        self.Omega_oil = Omega_oil

class PhysicalProperty:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)


class Pressure(PhysicalProperty):
    pass

class Viscosity(PhysicalProperty):
    pass

class Density(PhysicalProperty):
    pass

class Fraction(PhysicalProperty):
    pass


# R-22, R-115, будущее Oil (Iso32)
constants, properties = ChemicalConstantsPackage.from_IDs(['r-22', 'chloropentafluoroethane', 'water'])

# Свойства Oil (Молярная масса масла, температура (Кельвины), давление (Па), константа омега)
oil = Oil(400.0, 500, 950e3, 0.8)

# Перезаписали критические параметры для воды под масло ISO 32
constants2 = ChemicalConstantsPackage(MWs=constants.MWs[:-1] + [oil.M_oil], names=constants.names[:-1] + ['ISO 32'],
                                      omegas=constants.omegas[:-1] + [oil.Omega_oil], Pcs=constants.Pcs[:-1] + [oil.P_oil],
                                      Tcs=constants.Tcs[:-1] + [oil.T_oil])

# Массив температур
T = [-10, 0, 20, 40, 70, 80, 125]

# Давление
pressure_values = [[3.97, 3.8, 3.38, 3.05, 2.81, 2.24],
         [5.55, 4.85, 4.12, 3.17, 2.28],
         [10.14, 9.66, 5.14, 2.59],
         [16.767, 14.352, 10.695, 4.8645],
         [31.395, 25.3575, 18.699, 10.557, 2.553],
         [31.119, 25.9095, 7.59, 1.725],
         [32.20575, 18.94045, 8.142, 2.39755]]


# Вязкость R-22: от -10 градусов до 125
viscosity_values = [[20.286, 26.584, 48.671, 83.596, 123.132, 86.521],  # -10
             [12.905, 24.874, 55.281, 84.659, 101.942],  # 0
             [2.536, 4.706, 20.602, 48.295],  # 20
             [1.613, 36.5, 6.348, 11.599],  # 40
             [1.284, 1.959, 2.927, 4.53, 7.08],  # 70
             [1.863, 2.232, 3.46, 4.263],  # 80
             [1.492, 1.593, 1.877, 1.938]]  # 125

# Плотность R-22:
density_values = [[0.9708, 0.9687, 0.9494, 0.9384, 0.9265, 0.9255],  # -10
           [0.9728, 0.9539, 0.9258, 0.9169, 0.8962],  # 0
           [1.0286, 0.9827, 0.9124, 0.8889],  # 20
           [1.0027, 0.9487, 0.9183, 0.8747],  # 40
           [0.9835, 0.9391, 0.9087, 0.8653, 0.8414],  # 70
           [0.9176, 0.8925, 0.8399, 0.8259],  # 80
           [0.8785, 0.8488, 0.8273, 0.8073]]  # 125

# Доли R-22 в жидкой фазе
Fraction_R22_values = [[0.14240772, 0.13956069, 0.11894281, 0.09661578, 0.06626904, 0.06917368],
                [0.13752263, 0.097602225, 0.08366001, 0.0695826, 0.04349895],
                [0.21429768, 0.1465556, 0.0542013, 0.02756025],
                [0.214346975, 0.14201664, 0.08427174, 0.03830256],
                [0.19733655, 0.1429248, 0.10825416, 0.05892744, 0.01661184],
                [0.10424721, 0.0714132, 0.03954771, 0.04160995],
                [0.08350189, 0.05926864, 0.0315027, 0.01437675]]

# Доли R-115 в жидкой фазе
Fraction_R115_values = [[0.06045228, 0.05346931, 0.04211719, 0.04160422, 0.03917096, 0.03790632],
                 [0.09438737, 0.059947775, 0.04484999, 0.0370574, 0.02723105],
                 [0.13246232, 0.0898244, 0.0301587, 0.01326975],
                 [0.146203025, 0.09156336, 0.06461826, 0.02601744],
                 [0.12696345, 0.0803952, 0.05626584, 0.02995256, 0.01222816],
                 [0.05394279, 0.0361368, 0.02234229, 0.02134005],
                 [0.04320811, 0.02999136, 0.0177973, 0.00737325]]

pressure = Pressure(pressure_values)
viscosity = Viscosity(viscosity_values)
density = Density(density_values)
Fraction_R22 = Fraction(Fraction_R22_values)
Fraction_R115 = Fraction(Fraction_R115_values)

# Соотношение R-22 к R-115
# Общее соотношения газа к маслу с учетом распределительного коэффициента
g = 0.8
zs = [0.488 * g, 0.512 * g, 1 - g]

# Функция возврата мольных долей, температуры и давления
def calculate_flash(T, P, fractions, i, j):
    k13 = result.x[0] + (T[i] - T[0]) * result.x[1]
    k23 = result.x[2] + (T[i] - T[0]) * result.x[3]
    k12 = result.x[4] + (T[i] - T[0]) * result.x[5]

    kijs = [[0, k12, k13],
            [k12, 0, k23],
            [k13, k23, 0]]

    eos_kwargs = {'Pcs': constants2.Pcs, 'Tcs': constants2.Tcs, 'omegas': constants2.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs)
    flasher = FlashVL(constants2, properties, liquid=liquid, gas=gas)

    PT = flasher.flash(T=T[i] + 273.15, P=P[i][j] * pow(10, 5), zs=fractions)

    if PT.VF == 1:  # Есть только пар
        print('Неудача')
    else:  # Не только пар
        return [PT.liquid0.zs, T, P, PT.VF, k12, k13, k23]

# Функция оптимизации средней ошибки
def func_opt(parametrs):
    # Счетчик точек и суммы концентраций
    count = 0
    sum_R22 = 0
    sum_R115 = 0
    for i in range(len(T)):
        for j in range(len(pressure.values[i])):

            # kij значения для газов и масла
            k13 = parametrs[0] + (T[i] - T[0]) * parametrs[1]
            k23 = parametrs[2] + (T[i] - T[0]) * parametrs[3]
            k12 = parametrs[4] + (T[i] - T[0]) * parametrs[5]
            kijs = [[0, k12, k13],
                    [k12, 0, k23],
                    [k13, k23, 0]]

            eos_kwargs = {'Pcs': constants2.Pcs, 'Tcs': constants2.Tcs, 'omegas': constants2.omegas, 'kijs': kijs}
            gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs)
            liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs)
            flasher = FlashVL(constants2, properties, liquid=liquid, gas=gas)

            PT = flasher.flash(T=T[i] + 273.15, P=pressure.values[i][j] * pow(10, 5), zs=zs)
            if PT.VF != 1:  # Не только пар

                # Абсолютные ошибки
                ABS_error_R22 = abs(Fraction_R22.values[i][j] - PT.liquid0.zs[0])
                ABS_error_R115 = abs(Fraction_R115.values[i][j] - PT.liquid0.zs[1])

                # Счетчик точек
                count += 1
                sum_R22 += ABS_error_R22
                sum_R115 += ABS_error_R115

    # Средняя ошибка
    Avg_error_R22 = sum_R22 / count
    Avg_error_R115 = sum_R115 / count
    return Avg_error_R115 + Avg_error_R22

# ================= Оптимизация коэффициентов =================
# Оптимизируемые коэффициенты для R-22 и R-115
initial_guess_k13_k23 = [0.21, 0.001, 0.35, 0.001, 0.1, 0.001]
result = minimize(func_opt, initial_guess_k13_k23, bounds=((-1, 1), (0, 0.004), (-1, 1), (0, 0.004), (-1, 1), (0, 0.004)))
if result.success:
    print(result.message)
else:
    raise ValueError(result.message)

# Счетчик точек и суммы концентраций
count = 0
sum_R22 = 0
sum_R115 = 0

# Цикл расчета вспышки и ошибок
for i in range(len(T)):
    countT = 0
    sum_R22T = 0
    sum_R115T = 0
    # Массивы данных концентраций газов, давления
    R22_plot = []
    R115_plot = []
    P_bar_plot = []
    Fraction_R22_plot = []
    Fraction_R115_plot = []
    for j in range(len(pressure.values[i])):
        # Вызов функции для получения мольных долей, температуры и давления
        flash_data = calculate_flash(T, pressure.values, zs, i, j)
        # Добавление точек в листы
        # Реализуемая модель
        R22_plot.append(flash_data[0][0])
        R115_plot.append(flash_data[0][1])
        P_bar_plot.append(pressure.values[i][j])
        # Экспериментальные (исходные) данные
        Fraction_R22_plot.append(Fraction_R22.values[i][j])
        Fraction_R115_plot.append(Fraction_R115.values[i][j])

        # Абсолютные ошибки
        ABS_error_R22 = abs(Fraction_R22.values[i][j] - flash_data[0][0])
        ABS_error_R115 = abs(Fraction_R115.values[i][j] - flash_data[0][1])

        # Относительные ошибки
        Rel_error_R22 = ABS_error_R22/flash_data[0][0]
        Rel_error_R115 = ABS_error_R115 /flash_data[0][1]

        # Счетчик точек
        count += 1
        countT +=1
        sum_R22T += ABS_error_R22
        sum_R115T += ABS_error_R115

        sum_R22 += ABS_error_R22
        sum_R115 += ABS_error_R115

        print('==============================================================================')
        print('Температура расчета:', T[i], '°C; Давление расчета:', pressure.values[i][j], 'Бар')
        print('Соотношение компонентов смеси R-22, R-115, ISO 32:', flash_data[0])
        print('Пар ', flash_data[3], '\n')
        print('****** Абсолютная погрешность ******')
        print('Погрешность для R-22:', ABS_error_R22)
        print('Погрешность для R-115:', ABS_error_R115)
        print('****** Относительная погрешность ******')
        print('Погрешность для R-22:', Rel_error_R22)
        print('Погрешность для R-115:', Rel_error_R115)

    print('****** Значения kijs ******')
    print('k12=', flash_data[4])
    print('k13=', flash_data[5])
    print('k23=', flash_data[6])

    # Средняя ошибка для каждой температуры
    Avg_error_R22T = sum_R22T / countT
    Avg_error_R115T = sum_R115T / countT

    # Построение графика
    plt.title('График зависимости ω от P при Т='+str(T[i])+'°C', fontsize=18, fontname='Times New Roman')
    plt.xlabel('P, кПа', fontsize=18)
    plt.ylabel('ω, масс. доля', fontsize=18)
    plt.scatter(P_bar_plot, R22_plot)
    plt.scatter(P_bar_plot, R115_plot)
    plt.scatter(P_bar_plot, Fraction_R22_plot)
    plt.scatter(P_bar_plot, Fraction_R115_plot)
    plt.plot(P_bar_plot, R22_plot)
    plt.plot(P_bar_plot, R115_plot)
    plt.legend(['R-22 модел.', 'R-115 модел.', 'R-22 эксп.', 'R-115 эксп.'])
    plt.show()

# Средняя ошибка общая
Avg_error_R22 = sum_R22/count
Avg_error_R115 = sum_R115/count

print('****** Средняя погрешность ******')
print('Погрешность для R-22:', Avg_error_R22, '; В процентах:',round(Avg_error_R22*100,2),'%')
print('Погрешность для R-115:', Avg_error_R115, '; В процентах:',round(Avg_error_R115*100,2),'%')


