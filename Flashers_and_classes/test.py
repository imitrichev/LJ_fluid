from scipy.optimize import minimize
import numpy as np
import math
T = 333.15
R=8.314
gam1=[0.46, 0.4, 0.34, 0.38]
gam2=[0.9, 0.93, 0.93, 0.78]
x1=[0.7, 0.6, 0.5, 0.4]
arr1=[]
for i in range(0,4):
  arr1.append(gam1[i]*x1[i]+gam2[i]*(1-x1[i]))


def func(x,x1=x1,arr1=arr1):
  A=x[0]
  sum=0.0
  for i in range(0,4):
    sum+=abs(arr1[i]-A*x1[i]*(1-x1[i]))
  return sum


res = minimize(func, x0=[1.0], method='Nelder-Mead', tol=1e-6)
print(res.message)
print(res.x)
print(arr1)
for i in range(0, 4):
  print(res.x*x1[i]*(1-x1[i]))

import scipy.optimize as optimize

print('@@@@@@@@@@@')


def f(params):
  a, b, c = params
  return a ** 2 + b ** 2 + c ** 2


initial_guess = [1, 1, 1]
result = optimize.minimize(f, initial_guess)
if result.success:
  fitted_params = result.x
  print(fitted_params)
else:
  raise ValueError(result.message)