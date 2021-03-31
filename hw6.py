"""
hw6.py
Name(s):
NetId(s):
Date:
"""

import math
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

"""
FE: Forward Euler
"""
def FE(w0, z, m, w, x0, T, N):

	tn = 0
	def b(tn):
		b = np.array([[0],[math.cos(w*tn)/m]])
		return b

	A = np.array([[0,1],[-(w0)**2, -2*z*w0]])
	dt = T/N
	xn = np.zeros((2, N+1))
	xn[:,0,None] = x0
	t = [dt*n for n in range(N+1)]

	for n in range(N):
		tn = t[n]
		s = xn[:, n, None] + dt*np.matmul(A, xn[:,n,None]) + dt*b(tn)
		xn[:, n+1, None] = s

	x = xn[0]
	
	return (x,t)

"""
BE: Backward Euler
"""
def BE(w0, z, m, w, x0, T, N):

	tn = 0
	def b(tn):
		b = np.array([[0],[math.cos(w*tn)/m]])
		return b

	A = np.array([[0,1],[-(w0)**2, -2*z*w0]])
	dt = T/N
	xn = np.zeros((2, N+1))
	xn[:,0,None] = x0
	t = [dt*n for n in range(N+1)]

	dta = dt*A
	lhs = np.identity(2) - dta
	lu, piv = linalg.lu_factor(lhs)

	for n in range(N):
		tn = t[n]

		dtb = dt*b(tn+dt)
		rhs = xn[:, n, None] + dtb

		solution = linalg.lu_solve((lu,piv), rhs)
		xn[:, n+1, None] = solution

	x = xn[0]

	return (x,t)

"""
CN: Crank-Nicolson
"""
def CN(w0, z, m, w, x0, T, N):
	tn = 0
	def b(tn):
		b = np.array([[0],[math.cos(w*tn)/m]])
		return b

	A = np.array([[0,1],[-(w0)**2, -2*z*w0]])
	dt = T/N
	xn = np.zeros((2, N+1))
	xn[:,0,None] = x0
	t = [dt*n for n in range(N+1)]

	dta = 0.5 * dt * A
	lhs = np.identity(2) - dta
	lu, piv = linalg.lu_factor(lhs)

	for n in range(N):
		tn = t[n]

		rhs = np.matmul((np.identity(2) + 0.5*dt*A), xn[:,n,None]) + 0.5*dt*(b(tn+dt) + b(tn))

		solution = linalg.lu_solve((lu,piv),rhs)
		xn[:, n+1, None] = solution

	x = xn[0]
	return (x,t)

"""
RK4: fourth order Runge-Kutta
"""
def RK4(w0, z, m, w, x0, T, N):
	tn = 0
	def b(tn):
		b = np.array([[0],[math.cos(w*tn)/m]])
		return b

	A = np.array([[0,1],[-(w0)**2, -2*z*w0]])
	dt = T/N
	xn = np.zeros((2, N+1))
	xn[:,0,None] = x0
	t = [dt*n for n in range(N+1)]

	for n in range(N):
		tn = t[n]
		k1 = dt*(np.matmul(A, xn[:,n,None]) + b(tn))
		k2 = dt*(np.matmul(A, 0.5*k1+xn[:,n,None]) + b(tn + 0.5*dt))
		k3 = dt*(np.matmul(A, 0.5*k2+xn[:,n,None]) + b(tn + 0.5*dt))
		k4 = dt*(np.matmul(A, k3+xn[:,n,None]) + b(tn+dt))

		xn[:,n+1,None] = xn[:,n,None] + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4

	x = xn[0]

	return (x,t)

"""
main
"""
if __name__ == '__main__':

	# Testing Part 2: Numerical Methods

	# temp = np.array([[1],[1]])
	# disp, time = FE(1, 1, 1, 1, temp, 10, 10000)
	# print(disp)
	# print(time)

	x0 = np.array([[0],[0]])
	# disp, time = RK4(1, 1, 1, 1, x0, 10, 1000)
	# print(disp)
	# print(time)

	# Part 3: Testing the Methods/Error

	# solution = 0.5*(math.sin(10) - 10*math.exp(-10))

	N = [10**p for p in range(2,7)]
	# errorFE = [0 for p in range(5)]
	# errorBE = [0 for p in range(5)]
	# errorCN = [0 for p in range(5)]
	# errorRK4 = [0 for p in range(5)]

	# for i in range(len(N)):
	# 	n = N[i]
	# 	x,t = FE(1, 1, 1, 1, x0, 10, n)
	# 	errorFE[i] = abs(x[n] - solution)

	# 	x,t = BE(1, 1, 1, 1, x0, 10, n)
	# 	errorBE[i] = abs(x[n] - solution)

	# 	x,t = CN(1, 1, 1, 1, x0, 10, n)
	# 	errorCN[i] = abs(x[n] - solution)

	# 	x,t = RK4(1, 1, 1, 1, x0, 10, n)
	# 	errorRK4[i] = abs(x[n] - solution)

	# print(errorFE)
	# print(errorBE)
	# print(errorCN)
	# print(errorRK4)

	# Part 3: Plotting

	# errorFE = [7.88946562e-03, 7.50915688e-04, 7.47227998e-05, 7.46860367e-06, 7.46823616e-07]
	# errorBE = [7.07240584e-03, 7.42748873e-04, 7.46411320e-05, 7.46778710e-06, 7.46816556e-07]
	# errorCN = [5.77957025e-04, 5.77822211e-06, 5.77820765e-08, 5.77776715e-10, 6.07214279e-12]
	# errorRK4 = [1.06083735e-06, 1.01245956e-10, 9.99200722e-15, 1.22124533e-15, 2.52020627e-14]

	# plt.figure()
	# fig, ax = plt.subplots()
	# ax.plot(N, errorFE, label = "Error vs. N")
	# ax.set_xscale("log")
	# ax.set_yscale("log")
	# plt.title("Forward Euler: Error vs. N")
	# plt.xlabel("N (subintervals)")
	# plt.ylabel("Error")
	# plt.savefig("FE.png", bbox_inches = "tight")
	# plt.close("all")

	# plt.figure()
	# fig, ax = plt.subplots()
	# ax.plot(N, errorBE, label = "Error vs. N")
	# ax.set_xscale("log")
	# ax.set_yscale("log")
	# plt.title("Backward Euler: Error vs. N")
	# plt.xlabel("N (subintervals)")
	# plt.ylabel("Error")
	# plt.savefig("BE.png", bbox_inches = "tight")
	# plt.close("all")

	# plt.figure()
	# fig, ax = plt.subplots()
	# ax.plot(N, errorCN, label = "Error vs. N")
	# ax.set_xscale("log")
	# ax.set_yscale("log")
	# plt.title("Crank-Nicolson: Error vs. N")
	# plt.xlabel("N (subintervals)")
	# plt.ylabel("Error")
	# plt.savefig("CN.png", bbox_inches = "tight")
	# plt.close("all")

	# plt.figure()
	# fig, ax = plt.subplots()
	# ax.plot(N, errorRK4, label = "Error vs. N")
	# ax.set_xscale("log")
	# ax.set_yscale("log")
	# plt.title("Runge-Kutta 4: Error vs. N")
	# plt.xlabel("N (subintervals)")
	# plt.ylabel("Error")
	# plt.savefig("RK4.png", bbox_inches = "tight")
	# plt.close("all")

	# Part 4

	# x1, t = RK4(1, 0, 1, 0.8, x0, 100, 10000)
	# x2, t = RK4(1, 0, 1, 0.9, x0, 100, 10000)
	# x3, t = RK4(1, 0, 1, 1.0, x0, 100, 10000)

	# plt.figure()
	# fig, ax = plt.subplots()
	# ax.plot(t, x1, label = "x1 vs t")
	# plt.title("w = 0.8")
	# plt.xlabel("time")
	# plt.ylabel("x(t)")
	# plt.savefig("w1.png", bbox_inches = "tight")
	# plt.close("all")

	# plt.figure()
	# fig, ax = plt.subplots()
	# ax.plot(t, x2, label = "x1 vs t")
	# plt.title("w = 0.9")
	# plt.xlabel("time")
	# plt.ylabel("x(t)")
	# plt.savefig("w2.png", bbox_inches = "tight")
	# plt.close("all")

	# plt.figure()
	# fig, ax = plt.subplots()
	# ax.plot(t, x3, label = "x1 vs t")
	# plt.title("w = 1.0")
	# plt.xlabel("time")
	# plt.ylabel("x(t)")
	# plt.savefig("w3.png", bbox_inches = "tight")
	# plt.close("all")

	# Part 5

	wlist = [(0.1*p) for p in range(1,101)]
	maxlist = [0.0*p for p in range(100)]
	
	for i in range(100):
		w = wlist[i]
		x, t = RK4(1, 0.1, 1, w, x0, 100, 10000)

		maxx = abs(max(x))
		maxlist[i] = maxx

	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(wlist, maxlist, label = "max vs w")
	ax.set_xscale("log")
	ax.set_yscale("log")
	plt.title("Max Displacement vs. Forcing Frequency")
	plt.xlabel("Frequency (w)")
	plt.ylabel("Max Displacement")
	plt.savefig("frf.png", bbox_inches = "tight")
	plt.close("all")