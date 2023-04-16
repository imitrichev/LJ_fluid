from os import listdir


def aad(x_exp, x_calc):
    assert (len(x_exp) == len(x_calc))
    returnable = 0
    for i in range(len(x_exp)):
        returnable += abs((x_exp[i] - x_calc[i]) / x_exp[i])
    returnable *= 100 / len(x_exp)
    return returnable


def aad_new(params: list[float], func: callable, x: list, y: list[float], static_params: list[float] = None) -> float:
    if len(x) != len(y):
        raise ValueError(f"Input lists have different shapes: {len(x)} and {len(y)}")
    result = 0
    for i in range(len(x)):
        if static_params:
            result += abs((func(params, static_params, x[i]) - y[i]) / y[i])
        else:
            result += abs((func(params, x[i]) - y[i]) / y[i])
    result *= 100 / len(x)
    return result


def load_from_directory(dir_path: str, column: int) -> dict[float:list[float]]:
    result = {}
    for file in listdir(dir_path):
        with open(dir_path + "\\" + file, "r") as f:
            text = f.read()
            p = float(file[:-4])
            result[p] = {}
            for line in text.split('\n'):
                cur_line = line.split(' ')
                result[p][float(cur_line[0])] = float(cur_line[column])
    return result
