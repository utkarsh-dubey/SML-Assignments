import matplotlib.pyplot as plt
from math import *
import numpy as np

def func(n,x):
    return np.exp(-((x-n)**2)/2)/(2*np.pi)**(1/2)

x = np.linspace(-10,10,200)
px_w1 = func(2,x)
px_w2 = func(5,x)
px_w1_x_w2 = px_w1/px_w2

plt.figure(1)
plt.title("P(x/w1) vs x")
plt.plot(x,px_w1)
plt.xlabel("x")
plt.ylabel("P(x/w1)")
# plt.show()

plt.figure(2)
plt.title("P(x/w2) vs x")
plt.plot(x,px_w2)
plt.xlabel("x")
plt.ylabel("P(x/w2)")
# plt.show()

plt.figure(3)
plt.title("P(x/w1)/P(x/w2) vs x")
plt.plot(x,px_w1_x_w2)
plt.xlabel("x")
plt.ylabel("P(x/w1)/P(x/w2)")
plt.show()

