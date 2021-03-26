"""
hw6.py
Name(s):
NetId(s):
Date:
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class Matrix:

	"""
	Class attributes:
	mat:     the matrix itself, represented as a list of lists.
	numRows: the number of rows in the matrix.
	numCols: the number of columns in the matrix.
	L:       the lower triangular Matrix from the LU factorization.
	U:       the upper triangular Matrix from the LU factorization.
	P:       the permutation Matrix from the LU factorization.
	"""

	# Constructor method.
	def __init__(self, mat):
		self.mat = mat
		self.numRows = len(mat)
		self.numCols = len(mat[0])
		self.L = None
		self.U = None
		self.P = None

	# Special method used for printing this Matrix.
	# You will not have to alter this function.
	def __repr__(self):
		s = ''
		s += 'The %dx%d Matrix itself:\n\n' % (self.numRows, self.numCols)
		s += printMat(self.mat)
		s += '\n'
		if self.L != None:
			s += 'The lower triangular matrix L:\n\n'
			s += printMat(self.L.mat)
			s += '\n'
		if self.U != None:
			s += 'The upper triangular matrix U:\n\n'
			s += printMat(self.U.mat)
			s += '\n'
		if self.P != None:
			s += 'The permutation matrix P:\n\n'
			s += printMat(self.P.mat)
		return s

	"""
	matMult: multiplies two matrices.

	INPUT
	self, B: the Matrix-es being multiplied

	OUTPUT
	C: the Matrix result
	"""
	def matMult(self,B):
		# Get the dimensions of the (nxk).(kxm) multiplication.
		n = self.numRows
		k = self.numCols
		m = B.numCols

		# Ensure inner dimensions match.
		if k != B.numRows:
			raise(ValueError, 'Inner dimension mismatch.')

		# Initialize C as an nxm 2D list of floats.
		C = [[0.0 for x in range(m)] for y in range(n)]

		# Fill C in row major order.
		for r in range(n):
			for c in range(m):
				# Compute dot product of self[r][0:k] and B[0:k][c].
				for j in range(k):
					C[r][c] += self.mat[r][j]*B.mat[j][c]

		# Return the result as a Matrix.
		return Matrix(C)

	"""
	LUfact: performs LU factorization of self

	out: updates Matrix attributes L, U, and P
	"""
	def LUfact(self):
		# Get dimension and check that the Matrix is squre.
		n = self.numRows
		if n != self.numCols:
			raise(ValueError,'Matrix not square.')
		
		# Set some local pointers for ease and initialize matrices.
		# Setting the self.mat copy to A in order to follow the alg
		# from the slides.
		A = [self.mat[ind].copy() for ind in range(n)]
		P = [[0.0] * j + [1.0] + [0.0] * (n - 1 - j) for j in range(n)]
		L = [[0.0] * j + [1.0] + [0.0] * (n - 1 - j) for j in range(n)]
		
		# For each row of the matrix (excluding the last)...
		# Recall that the last row is at index n-1.
		for j in range(n-1):
			# Select the maximal pivot.
			pivotRow = j
			for i in range(j,n):
				# If new potential pivot is larger, update.
				if abs(A[i][j]) > abs(A[pivotRow][j]):
					pivotRow = i

			# Check for singular matrices.
			# Note: does not account for round off errors.
			if A[pivotRow][j] == 0:
				raise ValueError('Singular Matrix.')

			# Swap the rows of A, P, and L (just update pointers!).
			tempRow = A[j]
			A[j] = A[pivotRow]
			A[pivotRow] = tempRow
			tempRow = P[j]
			P[j] = P[pivotRow]
			P[pivotRow] = tempRow

			for k in range(len(L)):
				if(k < j and k < pivotRow):
					tempL = L[j][k]
					L[j][k] = L[pivotRow][k]
					L[pivotRow][k] = tempL

			# tempRow = L[j]
			# L[j] = L[pivotRow]
			# L[pivotRow] = tempRow

			# Elimination Step.
			for k in range(j+1,n):
				eLL = A[k][j]/A[j][j]
				L[k][j] = eLL
				for c in range(j,n):
					A[k][c] += -eLL*A[j][c]

		# Final check for singular matrices.
		# Note: does not account for round off errors.
		if A[-1][-1] == 0:
			raise ValueError('Singular Matrix.')

		# Convert U, L, and P into Matrix objects and return.
		self.U = Matrix(A)
		self.P = Matrix(P)
		self.L = Matrix(L)
		return

	"""
	backSub: performs the backward substitution step using self.U

	INPUT
	c: the RHS column vector, a Matrix object.

	OUTPUT
	x: the resulting colmn vector, a Matrix object.
	"""
	def backSub(self,c):
		# Create local pointer so we don't have to retype self.U alot.
		U = self.U
		
		# Preallocate x as a column vector of the correct size.
		n = len(c.mat)
		x = [[0.0] for j in range(n)]

		# Fill in the last entry of x.
		x[-1][0] = c.mat[-1][0]/U.mat[-1][-1]

		# Loop backwards, skipping LAST row.
		for i in range(n-1):
			j = n - 2 - i

			# Compute the summation term.
			s = 0
			for k in range(j+1,n):
				s += U.mat[j][k]*x[k][0]

			# Update jth entry in x.
			x[j][0] = (c.mat[j][0] - s)/U.mat[j][j]

		# Return the result as a Matrix.
		return Matrix(x)

	"""
	forSub: performs the forward substitution step using self.L

	INPUT
	b: the RHS column vector, a Matrix object.

	OUTPUT
	c: the resulting colmn vector, a Matrix object.
	"""
	def forSub(self,b):
		# Get the relevent dimension.
		n = self.numCols

		# Create local pointer so we don't have to retype self.L alot.
		L = self.L
		
		# Preallocate c as a column vector of the correct size.
		c = [[0.0] for j in range(n)]

		# Fill in the first entry of c.
		c[0][0] = b.mat[0][0]

		# Loop through the other rows.
		for j in range(1,n):
			# Compute the summation term.
			s = 0
			for k in range(0, j):
				s += L.mat[j][k]*c[k][0]

			# Update jth entry in c.
			c[j][0] = b.mat[j][0] - s

		# Return the result.
		return Matrix(c)

	"""
	gaussElim: performs Gaussian Elimination to solve Ax=b.

	INPUT
	(self): the Matrix A.
	b: the RHS, as a column vector Matrix object.

	OUTPUT
	x: the resulting column vector Matrix object.
	"""
	def gaussElim(self,b):
		# Check dimensions.
		if self.numRows != b.numRows:
			raise(ValueError,'Dimension mismatch.')

		# Gaussian Elimination
		self.LUfact()            # Perform the LU factorization.
		bHat = self.P.matMult(b) # Multiply bHat = P*b.
		c = self.forSub(bHat)    # Solve Lc = bHat.
		x = self.backSub(c)      # Solve Ux = c.
		return x

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
	t = [[dt*n for n in range(N+1)]]
	t = np.array(t)

	for n in range(N):
		tn = t[0][n]
		s = xn[:, n, None] + dt*np.matmul(A, xn[:,n,None]) + dt*b(tn)
		xn[:, n+1, None] = s

	x = np.array([xn[0]])
	
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
	t = [[dt*n for n in range(N+1)]]
	t = np.array(t)

	for n in range(N):
		tn = t[0][n]
		dta = dt * A
		lhs = np.identity(2) - dta

		dtb = dt*b(tn+dt)
		rhs = xn[:, n, None] + dtb

		lhs = lhs.tolist()
		lhs = Matrix(lhs)
		rhs = rhs.tolist()
		rhs = Matrix(rhs)

		solution = lhs.gaussElim(rhs)
		solution = solution.mat
		xn[:, n+1, None] = solution

	x = np.array([xn[0]])

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
	t = [[dt*n for n in range(N+1)]]
	t = np.array(t)

	for n in range(N):
		tn = t[0][n]
		dta = 0.5 * dt * A
		lhs = np.identity(2) - dta

		rhs = np.matmul((np.identity(2) + 0.5*dt*A), xn[:,n,None]) + 0.5*dt*(b(tn+dt) + b(tn))

		lhs = lhs.tolist()
		lhs = Matrix(lhs)
		rhs = rhs.tolist()
		rhs = Matrix(rhs)

		solution = lhs.gaussElim(rhs)
		solution = solution.mat
		xn[:, n+1, None] = solution

	x = np.array([xn[0]])
	return (x,t)

"""
RK4: fourth order Runge-Kutta
"""
def RK4(w0, z, m, w, x0, T, N):
	return (x,t)

"""
main
"""
if __name__ == '__main__':
	# temp = np.array([[1],[1]])
	# disp, time = FE(1, 1, 1, 1, temp, 10, 10000)
	# print(disp)
	# print(time)

	temp = np.array([[0],[0]])
	disp, time = CN(1, 1, 1, 1, temp, 10, 10000)
	print(disp)
	print(time)
