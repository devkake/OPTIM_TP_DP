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

def K2(x):
	return -x

def K3(V, x):
	return V[0, x]

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

def MC1(x0, ufix, change=True):
	z = 0
	x = x0
	if change:
		for n in nl:
			# order is very important
			tx = x
			z = z - cost * min(ufix, xmax - tx) # difference
			tx = tx + min(ufix, xmax - tx)
			nc = np.random.binomial(n, p)
			z = z + price * min(nc, tx)
			tx = tx - min(nc, tx)
			x = tx
	else:
		for n in nl:
			# order is very important
			tx = x
			z = z - cost * ufix # difference
			tx = tx + min(ufix, xmax - tx)
			nc = np.random.binomial(n, p)
			z = z + price * min(nc, tx)
			tx = tx - min(nc, tx)
			x = tx
	return z

def MC2(x0, pi, change=True):
	z = 0
	x = x0
	if change:
		for i in range(len(nl)):
			# order is very important
			n = nl[i]
			tx = x
			us = pi[i][tx]
			z = z - cost * min(us, xmax - x) # Difference
			tx = tx + min(us, xmax - x) 
			nc = np.random.binomial(n, p)
			z = z + price * min(nc, tx)
			tx = tx - min(nc, tx)
			x = tx
	else:
		for i in range(len(nl)):
			# order is very important
			n = nl[i]
			tx = x
			us = pi[i][tx]
			z = z - cost * us # Difference
			tx = tx + min(us, xmax - x) 
			nc = np.random.binomial(n, p)
			z = z + price * min(nc, tx)
			tx = tx - min(nc, tx)
			x = tx
	return z

if __name__ == '__main__':

	print("dynamic programming start")

	np.random.seed()

	print("Question 3")
	
	V = np.zeros((T + 1, xmax - xmin + 1)) 
	pi = np.zeros((T, xmax - xmin + 1)) # strategy 

	for i in range(xmax - xmin + 1):
		V[T, i] = K(i)

	for t in range(T - 1, -1, -1):
		n = nl[t]
		for x in range(xmax - xmin + 1):
			V[t, x] = np.inf
			for u in range(0, min(xmax + 1 - x, umax + 1)):
				vu = E_L(x, u, n) + E_V(x, u, t, V)
				if vu < V[t, x]:
					V[t, x] = vu
					pi[t, x] = u
	
	# print(V)
	# print(optimal(V))
	print(pi)

	opt = -V[0]
	plt.plot(opt)
	plt.title("Initial Stock vs. Optimal Value (=V(0))")
	plt.show()

	print("Question 4")

	precost1 = 0.75
	precost2 = 1.25

	precost = 1

	preopt = np.zeros(opt.shape)
	bought = np.zeros(opt.shape)
	after = np.zeros(opt.shape)

	for i in range(xmax - xmin + 1):
		tl = np.array([opt[j] - precost * max(0, j - i) for j in range(xmax - xmin + 1)])
		preopt[i] = max(tl)
		bought[i] = max(0, np.argmax(tl) - i)
		after[i] = i + bought[i]

	plt.plot(preopt)
	plt.title("Initial Stock vs. Optimal Value with Precost = " + str(precost))
	plt.show()
	plt.plot(bought)
	plt.title("Initial Stock vs. Items bought with Precost = " + str(precost))
	plt.show()
	plt.plot(after)
	plt.title("Initial Stock vs. After Stock with Precost = " + str(precost))
	plt.show()

	print("Question 5")

	x0 = 15
	m = 3000 # the number of sampleing for Monte-Carlo
	tu = 30
	ufl = list(range(0, tu + 1))
	vufl = (tu + 1) * [0] 

	for i in range((tu + 1)):
		mel = m * [0]
		ufix = ufl[i]
		for j in range(m):
			mel[j] = MC1(x0, ufix, change=True) # strategy can be changed in the week
		vufl[i] = np.mean(mel)

	plt.plot(vufl)
	plt.title("Fixed Product vs. Profit Expected with Initial Stock  = " + str(x0))
	plt.show()

	print("Question 6")

	x0 = 15
	m = 10000
	vl = (m + 1) * [0]

	for i in range(m):
		vl[i] = MC2(x0, pi, change=True) # strategy can be changed in the week

	print("Optimal = " + str(-V[0][x0]))
	print("MC = " + str(np.mean(vl)))

	print("Question 7")
	
	V2 = np.zeros((T + 1, xmax - xmin + 1)) 
	pi2 = np.zeros((T, xmax - xmin + 1)) # strategy 

	for i in range(xmax - xmin + 1):
		V2[T, i] = K2(i)

	for t in range(T - 1, -1, -1):
		n = nl[t]
		for x in range(xmax - xmin + 1):
			V2[t, x] = np.inf
			for u in range(0, min(xmax + 1 - x, umax + 1)):
				vu = E_L(x, u, n) + E_V(x, u, t, V2)
				if vu < V2[t, x]:
					V2[t, x] = vu
					pi2[t, x] = u
	
	# print(V)
	# print(optimal(V))
	print(pi2)

	opt2 = -V2[0]
	plt.plot(opt, label="K(x)=0")
	plt.plot(opt2, label="K(x)=-x")
	plt.title("Initial Stock vs. Optimal Value (=V(0)) Comparison")
	plt.show()

	print("Question 8")

	V3 = np.zeros((T + 1, xmax - xmin + 1)) 
	pi3 = np.zeros((T, xmax - xmin + 1)) # strategy 

	for i in range(xmax - xmin + 1):
		V3[T, i] = K3(V, i)

	for t in range(T - 1, -1, -1):
		n = nl[t]
		for x in range(xmax - xmin + 1):
			V3[t, x] = np.inf
			for u in range(0, min(xmax + 1 - x, umax + 1)):
				vu = E_L(x, u, n) + E_V(x, u, t, V3)
				if vu < V3[t, x]:
					V3[t, x] = vu
					pi3[t, x] = u
	
	# print(V3)
	# print(optimal(V3))
	print(pi3)

	opt3 = -V3[0]
	plt.plot(opt3)
	plt.title("Initial Stock vs. Optimal Value (=V(0)) for 2 weeks")
	plt.show()

	print("Question 9")

	# just calculate another DP programming 












