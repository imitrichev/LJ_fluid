from os import listdir


def load_from_directory(dir_path: str, column: int) -> dict[float:dict[float:float]]:
    result = {}
    for file in listdir(dir_path):
        with open(dir_path + "\\" + file, "r") as f:
            text = f.read()
            # Считываем давление из имени
            p = float(file[:-4])
            result[p] = {}
            for line in text.split('\n'):
                cur_line = line.split(' ')
                # Забираем значение из искомой колонки
                result[p][float(cur_line[0])] = float(cur_line[column])
    return result


class ViscData:
    def __init__(self, dir_path: str):
        self.density = load_from_directory(dir_path, 2)
        self.visc_k = load_from_directory(dir_path, 3)
        self.visc_d = load_from_directory(dir_path, 4)


def aad(x_exp, x_calc):
    assert (len(x_exp) == len(x_calc))
    returnable = 0
    for i in range(len(x_exp)):
        returnable += abs((x_exp[i] - x_calc[i]) / x_exp[i])
    returnable *= 100 / len(x_exp)
    return returnable


def aad_new(params: list[float], func: callable, points: list, exp: list[float],
            static_params: list[float] = None) -> float:
    if len(points) != len(exp):
        raise ValueError(f"Input lists have different shapes: {len(points)} and {len(exp)}")
    result = 0
    for i in range(len(points)):
        if static_params:
            result += abs((func(params, static_params, points[i]) - exp[i]) / exp[i])
        else:
            try:
                result += abs((func(params, points[i]) - exp[i]) / exp[i])
            except KeyError:
                pass
    result *= 100 / len(points)
    return result
