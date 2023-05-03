# Задача. Предсказывание состава жидкой фазы смеси.
# Входные параметры: Температура C, давление Бар (МПа/10) для каждой точки, вязкость cP, плотность г/мл и состав газа
# R-22/R-115 (48.8/51.2) ------> (0.488/0.512)
# Построить зависимость концентрации от давления

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, FlashVL


def read_data_from_file_lines(file_path):  # Функция считывания данных из файла
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 1:
            data = [float(x) for x in lines[0].strip().split(',')]
        else:
            for line in lines:
                values = line.strip().split(',')
                data.append([float(v) for v in values])
    return data


# Функция возврата мольных долей, температуры и давления
def calculate_flash(T, P, fractions, i, j):
    k13 = result.x[0] + (T[i] - T[0]) * result.x[1]
    k23 = result.x[2] + (T[i] - T[0]) * result.x[3]
    k12 = result.x[4] + (T[i] - T[0]) * result.x[5]

    kijs = [[0, k12, k13],
            [k12, 0, k23],
            [k13, k23, 0]]

    eos_kwargs = {'Pcs': constants2.Pcs, 'Tcs': constants2.Tcs, 'omegas': constants2.omegas, 'kijs': kijs}
    gas = CEOSGas(SRKMIX, eos_kwargs=eos_kwargs)
    liquid = CEOSLiquid(SRKMIX, eos_kwargs=eos_kwargs)
    flasher = FlashVL(constants2, properties, liquid=liquid, gas=gas)

    PT = flasher.flash(T=T[i] + 273.15, P=P[i][j] * pow(10, 5), zs=fractions)

    if PT.VF == 1:  # Есть только пар
        return [PT.gas.zs, T, P, PT.VF, k12, k13, k23]
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
            gas = CEOSGas(SRKMIX, eos_kwargs=eos_kwargs)
            liquid = CEOSLiquid(SRKMIX, eos_kwargs=eos_kwargs)
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


class Temperature(PhysicalProperty):
    pass


class Pressure(PhysicalProperty):
    pass


class Viscosity(PhysicalProperty):
    pass


class Density(PhysicalProperty):
    pass


class Fraction(PhysicalProperty):
    pass


# Параметры: массив температур
T = read_data_from_file_lines('../data/temperature.txt')

# Давление R-22: от -10 градусов до 125
pressure_values = read_data_from_file_lines('../data/pressure.txt')

# Вязкость R-22: от -10 градусов до 125
viscosity_values = read_data_from_file_lines('../data/viscosity.txt')

# Плотность R-22:
density_values = read_data_from_file_lines('../data/density.txt')

# Доли R-22 в жидкой фазе
Fraction_R22_values = read_data_from_file_lines('../data/fraction_r22.txt')

# Доли R-115 в жидкой фазе
Fraction_R115_values = read_data_from_file_lines('../data/fraction_r115.txt')

# R-22, R-115, Oil (Iso32)
constants, properties = ChemicalConstantsPackage.from_IDs(['r-22', 'chloropentafluoroethane', 'water'])

# Критические параметры масла:
M_oil = 400.0  # Молярная масса масла
T_oil = 800  # Кельвины
P_oil = 950e3  # Па
Omega_oil = 0.2  # Константа омега

# Перезаписали критические параметры для воды под масло ISO 32
constants2 = ChemicalConstantsPackage(MWs=constants.MWs[:-1] + [M_oil], names=constants.names[:-1] + ['ISO 32'],
                                      omegas=constants.omegas[:-1] + [Omega_oil], Pcs=constants.Pcs[:-1] + [P_oil],
                                      Tcs=constants.Tcs[:-1] + [T_oil])

temperature = Temperature(T)
pressure = Pressure(pressure_values)
viscosity = Viscosity(viscosity_values)
density = Density(density_values)
Fraction_R22 = Fraction(Fraction_R22_values)
Fraction_R115 = Fraction(Fraction_R115_values)

# Общее соотношения газа к маслу (распределительный коэффициент)
g = 0.8
# Соотношение R-22 к R-115 с маслом
zs = [0.488 * g, 0.512 * g, 1 - g]

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
        if flash_data[3] == 1:
            print('================ПРИСУТСТВУЕТ ТОЛЬКО ПАР================')
            print('Температура расчета:', T[i], '°C; Давление расчета:', pressure_values[i][j], 'Бар')
            print('Соотношение компонентов смеси в газе R-22, R-115, ISO 32:', flash_data[0])
            print('Пар ', flash_data[3], '\n')
        else:
            # Добавление точек в листы
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
            print('Пар ', flash_data[3])
            print('****** Абсолютная погрешность ******')
            print('Погрешность для R-22:', ABS_error_R22)
            print('Погрешность для R-115:', ABS_error_R115)
            print('****** Относительная погрешность ******')
            print('Погрешность для R-22:', Rel_error_R22)
            print('Погрешность для R-115:', Rel_error_R115, '\n')

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
