import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def real_func(x):
    return np.sin(2 * np.pi * x)



def fit_func(p, x):
    f = np.poly1d(p)  #数字1不是ld ;单行注释。
    return f(x)


def residuals_func(p, x, y):
    ret = fit_func(p,x) - y
    return ret


x = np.linspace(0, 1, 10)
print(x)
x_points = np.linspace(0, 1, 1000)
print('----------')
print(x_points)
print('------------')
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

def fitting1(M=0):
    p_init=np.random.rand(M+1)
    p_lsq=leastsq(residuals_func,p_init,args=(x,y))
    print('Fitting Parameters:',p_lsq[0])

    plt.plot(x_points,real_func(x_points),label='real')
    plt.plot(x_points,fit_func(p_lsq[0],x_points),label='fitting curve')
    plt.plot(x,y,'bo',label='noise')
    plt.legend()
    return p_lsq
#M=0
plt.text(0,0.9,"M=0")
p_lsq_0=fitting1(M=0)

#M=1
plt.text(0,0.9,"M=1")
p_lsq_1=fitting1(M=1)

#M=3
plt.text(0,0.9,"M=3")
p_lsq_3=fitting1(M=3)

#M=9
plt.text(0,0.9,"M=9")
#fitting1(M=9)
p_lsq_9 = fitting1(M=9)

regularization=0.0001
def residuals_func_regularization(p,x,y):
    ret=fit_func(p,x)-y
    ret=np.append(ret,np.sqrt(0.5*regularization*np.square(p)))
    return ret

p_int=np.random.rand(9+1)
p_lsq_regularization=leastsq(residuals_func_regularization,p_int,args=(x,y))

plt.plot(x_points,real_func(x_points),label='real')
plt.plot(x_points,fit_func(p_lsq_9[0],x_points),label='fitted curve')
plt.plot(x_points,fit_func(p_lsq_regularization[0],x_points),label='regularization')
plt.plot(x,y,'bo',label='noise')
plt.legend()
