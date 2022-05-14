import matplotlib.pyplot as plt
from math import *
import numpy as np


def pdf(a,b,x):
    return 1/(2*np.pi*b*(1+((x-a)/b)**2))

x = np.linspace(-10,10,200)
pw1_x = pdf(3,1,x) / (pdf(3,1,x) + pdf(5,1,x))

plt.title("p(w1/x) vs x plot")
plt.plot(x,pw1_x)
plt.xlabel("x")
plt.ylabel("p(w1/x)")
plt.show()