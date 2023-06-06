import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

A = pd.read_excel("Water 1 27 Measurement.xlsx")
x= A[A.columns[372]].values.tolist()
y= A[A.columns[373]].values.tolist()
print(x,y)
n=300

def r_square(k):
    i=0
    rk=0
    while i<(n-k):
          rk += ((x[i+k]-x[i])**2+(y[i+k]-y[i])**2)/(n-k)
          i=i+1
    return rk

r = []
j=1
while j<n:
    r.append(r_square(j))
    j=j+1

t = list(range(1,n))
plt.plot(t,r,".b")

def func(x, a):
    return a*x
def fit(func, x, y):
    popt, pcov = curve_fit(func, x,y)
    return popt[0], pcov
x2 = np.linspace(t[0], t[298], num=50)
fit00 = fit(func, t, r)
plt.plot(x2, func(x2, fit00[0]), "-b",label="2D")
plt.xlabel("Time")
plt.ylabel("<r^2>")
plt.suptitle("27°C")
plt.title("<r^2> versus Time")
print(fit00)

def r_square_x(k):
    i=0
    rk=0
    while i<(n-k):
          rk += ((x[i+k]-x[i])**2)/(n-k)
          i=i+1
    return rk

r_x = []
j=1
while j<n:
    r_x.append(r_square_x(j))
    j=j+1

plt.plot(t,r_x,".r")

def func(x, a):
    return a*x
def fit(func, x, y):
    popt, pcov = curve_fit(func, x,y)
    return popt[0], pcov
x2 = np.linspace(t[0], t[298], num=50)
fit01 = fit(func, t, r_x)
plt.plot(x2, func(x2, fit01[0]), "-r",label="x_direction")
print(fit01)

def r_square_y(k):
    i=0
    rk=0
    while i<(n-k):
          rk += ((y[i+k]-y[i])**2)/(n-k)
          i=i+1
    return rk

r_y = []
j=1
while j<n:
    r_y.append(r_square_y(j))
    j=j+1

plt.plot(t,r_y,".g")

def func(x, a):
    return a*x
def fit(func, x, y):
    popt, pcov = curve_fit(func, x,y)
    return popt[0], pcov
x2 = np.linspace(t[0], t[298], num=50)
fit02 = fit(func, t, r_y)
plt.plot(x2, func(x2, fit02[0]), "-g",label="y-direction")
print(fit02)
plt.legend(loc="upper left")

plt.savefig('27°C x and y.png')