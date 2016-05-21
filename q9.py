import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

import pickle

global T

global x1min
global x1max
global u1max
global cost1
global price1
global p1
global nl1

global x2min
global x2max
global u2max
global cost2
global price2
global p2
global nl2

T = 7

xmin1 = 0
xmax1 = 50
umax1 = 10
cost1 = 1
price1 = 2
p1 = 0.5
nl1 = [15, 12, 10, 10, 10, 40, 40]

xmin2 = 0
xmax2 = 60
umax2 = 20
cost2 = 2
price2 = 3
p2 = 0.4
nl2 = [20, 20, 25, 15, 35, 45, 30]

def f(x, u, w):
	return max(x + u - w, 0)

def profit1(x, u, w):
	return price1 * min(w, x + u) - cost1 * u

def profit2(x, u, w):
	return price2 * min(w, x + u) - cost2 * u

def L1(x, u, w):
	return -profit1(x, u, w)

def L2(x, u, w):
	return -profit2(x, u, w)

def K(x):
	return 0

def E_L1(x, u, n):
	wl = list(range(0, n+1))
	wpl = np.array(binom.pmf(wl, n, p1))
	ll = np.array([L1(x, u, w) for w in wl])
	el = sum(wpl * ll)
	return el

def E_L2(x, u, n):
	wl = list(range(0, n+1))
	wpl = np.array(binom.pmf(wl, n, p2))
	ll = np.array([L2(x, u, w) for w in wl])
	el = sum(wpl * ll)
	return el

def E_V1(x, u, t, V):
	wl = list(range(0, n+1))
	wpl = np.array(binom.pmf(wl, n, p1))
	xl = [f(x, u, w) for w in wl]
	vl = np.array([V[t + 1, xn] for xn in xl])
	ev = sum(wpl * vl)
	return ev

def E_V2(x, u, t, V):
	wl = list(range(0, n+1))
	wpl = np.array(binom.pmf(wl, n, p2))
	xl = [f(x, u, w) for w in wl]
	vl = np.array([V[t + 1, xn] for xn in xl])
	ev = sum(wpl * vl)
	return ev

if __name__ == '__main__':

	print("dynamic programming start")

	np.random.seed()

	print("Question 9")
	
	V1 = np.zeros((T + 1, xmax1 - xmin1 + 1)) 
	pi1 = np.zeros((T, xmax1 - xmin1 + 1)) # strategy 

	for i in range(xmax1 - xmin1 + 1):
		V1[T, i] = K(i)

	for t in range(T - 1, -1, -1):
		n = nl1[t]
		for x in range(xmax1 - xmin1 + 1):
			V1[t, x] = np.inf
			for u in range(0, min(xmax1 + 1 - x, umax1 + 1)):
				vu = E_L1(x, u, n) + E_V1(x, u, t, V1)
				if vu < V1[t, x]:
					V1[t, x] = vu
					pi1[t, x] = u
	
	V2 = np.zeros((T + 1, xmax2 - xmin2 + 1)) 
	pi2 = np.zeros((T, xmax2 - xmin2 + 1)) # strategy 

	for i in range(xmax2 - xmin2 + 1):
		V2[T, i] = K(i)

	for t in range(T - 1, -1, -1):
		n = nl2[t]
		for x in range(xmax2 - xmin2 + 1):
			V2[t, x] = np.inf
			for u in range(0, min(xmax2 + 1 - x, umax2 + 1)):
				vu = E_L2(x, u, n) + E_V2(x, u, t, V2)
				if vu < V2[t, x]:
					V2[t, x] = vu
					pi2[t, x] = u

	# print(V)
	# print(optimal(V))
	# print(pi)

	opt1 = -V1[0]
	plt.plot(opt1)
	plt.title("Initial Stock vs. Optimal Value (=V(0)) [Product 1]")
	plt.show()

	opt2= -V2[0]
	plt.plot(opt2)
	plt.title("Initial Stock vs. Optimal Value (=V(0)) [Product 2]")
	plt.show()