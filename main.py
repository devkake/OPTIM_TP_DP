import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

import pickle


global xmin
global xmax
global umax
global T
global cost
global price
global p
global nl

xmin = 0
xmax = 50
umax = 10
T = 7
cost = 1
price = 2
p = 0.5
nl = [15, 12, 10, 10, 10, 40, 40]

def f(x, u, w):
	return max(x + u - w, 0)

def profit(x, u, w):
	return price * min(w, x + u) - cost * u

def L(x, u, w):
	return -profit(x, u, w)

def K(x):
	return 0

def E_L(x, u, n):
	wl = list(range(0, n+1))
	wpl = np.array(binom.pmf(wl, n, p))
	ll = np.array([L(x, u, w) for w in wl])
	el = sum(wpl * ll)
	return el

def E_V(x, u, t, V):
	wl = list(range(0, n+1))
	wpl = np.array(binom.pmf(wl, n, p))
	xl = [f(x, u, w) for w in wl]
	vl = np.array([V[t + 1, xn] for xn in xl])
	ev = sum(wpl * vl)
	return ev

def optimal(V):
	return -min(V[0])

if __name__ == '__main__':
	print("dynamic programming start")

	print("Question 3")
	
	V = np.zeros((T + 1, xmax - xmin + 1))
	pi = np.zeros((T + 1, xmax - xmin + 1))

	for i in range(xmax - xmin + 1):
		V[T, i] = K(i)

	for t in range(T - 1, -1, -1):
		n = nl[t]
		for x in range(xmin, xmax + 1):
			V[t, x] = np.inf
			for u in range(0, min(xmax + 1 - x, umax + 1)):
				vu = E_L(x, u, n) + E_V(x, u, t, V)
				if vu < V[t, x]:
					V[t, x] = vu
					pi[t, x] = u
	
	print(V)
	print(optimal(V))
	print(pi)

	print("Question 4")