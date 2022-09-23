from numpy import multiply, exp, negative, divide, log, subtract, empty


def viscosity(x, mw_i, nu_i, t):
    kb = 1.380649 * 1e-23
    na = 6.022 * 1e23
    tc_ii = 516.25
    tc_jj = 540.61
    vc_ii = 182.49 * 1e-6
    vc_jj = 428.64 * 1e-6
    eps_ii = kb * tc_ii / 1.2593
    eps_jj = kb * tc_jj / 1.2593
    sigma_ii = (0.302 * vc_ii / na) ** (1 / 3)
    sigma_jj = (0.302 * vc_jj / na) ** (1 / 3)
    sigma_ij = ((sigma_ii ** 3 + sigma_jj ** 3) / 2) ** (1 / 3)
    eps_ij = (eps_ii * sigma_ii ** 3 + eps_jj * sigma_jj ** 3) / 2 / sigma_ij ** 3
    sigma = [[sigma_ii, sigma_ij],
             [sigma_ij, sigma_jj]]
    eps = [[eps_ii, eps_ij],
           [eps_ij, eps_jj]]
    sigma_x = 0.0
    eps_x = 0.0
    nu = 0.0
    mwx = 0.0
    for i in range(0, 2):
        nu += nu_i[i] * x[i]
        mwx += mw_i[i] * x[i]
        for j in range(0, 2):
            sigma_x += x[i] * x[j] * sigma[j][i] ** 3
            eps_x += x[i] * x[j] * eps[j][i] * sigma[i][j] ** 3
    sigma_x = sigma_x ** (1 / 3)
    eps_x /= sigma_x ** 3
    t_ast = kb * t / eps_x
    rho_ast = na * (sigma_x ** 3) / nu
    coeff = [1.06036, 0.15610, 0.19300, 0.47635, 1.03587, 1.52996, 1.76474, 3.89411]
    b = [0.062692, 4.095577, -8.743269 * 1e-6, 11.124920, 2.542477 * 1e-6, 14.863984]
    collision_integral = coeff[0] / t_ast ** coeff[1] + coeff[2] / exp(coeff[3] * t_ast) + coeff[4] / exp(
        coeff[5] * t_ast) + coeff[6] / exp(coeff[7] * t_ast)
    ac = 0.95
    visc_0_ast = 0.17629 * t_ast ** (1 / 2) * ac / collision_integral
    delta_visc_ast = b[0] * (exp(b[1] * rho_ast) - 1) + b[2] * (exp(b[3]) * rho_ast) + b[4] * t_ast ** (
                -2 ** exp(b[5] * rho_ast) - 1)
    visc_ast = visc_0_ast+delta_visc_ast
    visc = visc_ast*(mwx*eps_x/na)**0.5/(sigma_x**2)
    return visc


vexp = [
    0.410, 0.419, 0.450, 0.494, 0.556, 0.644, 0.757, 0.931, 1.194,
    0.509, 0.518, 0.547, 0.597, 0.668, 0.760, 0.885, 1.052, 1.344,
    0.604, 0.614, 0.648, 0.707, 0.787, 0.888, 1.023, 1.200, 1.494,
    0.706, 0.718, 0.757, 0.824, 0.913, 1.022, 1.168, 1.354, 1.647,
    0.817, 0.832, 0.875, 0.951, 1.047, 1.162, 1.316, 1.509, 1.797,
    0.937, 0.953, 1.003, 1.087, 1.191, 1.312, 1.473, 1.674, 1.955]

for i in range(len(vexp)):
    vexp[i] /= 1000
T = 293.15
P = [0.1, 20, 40, 60, 80, 100]
Mw = [46.07 / 1000, 100.2 / 1000]
constC = 0.0894
B0 = [1e2, 1e2]
B1 = [0.0009, 0.0009]
alpha_n = [0.002, 0.004]
nu0 = [1.04958e-4, 4.76409e-4]
nuP0 = multiply(nu0, exp(multiply(negative(alpha_n), T)))
B = multiply(B0, exp(multiply(negative(B1), T)))

vvv = empty((len(P), len(vexp)))

for j in range(0, len(P)):
    nu = multiply(nuP0, subtract(1, multiply(constC, log(multiply(1 + P[j], (divide(1, B)))))))
    x = [0, 0]
    for i in range(0, len(vexp)):
        x[0] = i*0.125
        x[1] = 1-x[0]
        vvv[j][i] = viscosity(x, Mw, nu, T)
