# -*- coding: UTF-8 -*-

import copy
import numpy as np

import sys
sys.path.append('./scikit-tensor')
from sktensor import dtensor, tucker_hooi

def compute_loss(A, R):
	return (((A-R)*(A>10))**2).sum()

def Gradient_Tensor_Decomposition(A, K, steps=500, alpha=0.0002, beta=0.02, epsilon=1):
	# Initialization
	len_X, len_Y, len_Z = A.shape
	print len_X, len_Y, len_Z
	C = dtensor(np.random.rand(K,K,K))
	X = np.random.rand(len_X,K)
	Y = np.random.rand(len_Y,K)
	Z = np.random.rand(len_Z,K) 
	# Iteration
	R = C.ttm(X, 0).ttm(Y, 1).ttm(Z, 2)
	loss_last, loss_curr, step = float('inf'), compute_loss(A,R), 0
	for step in xrange(steps):
		print step
		for dim1 in xrange(len_X):
			print dim1
			for dim2 in xrange(len_Y):
				for dim3 in xrange(len_Z):
					loss_last = loss_curr
					_X, _Y, _Z, _C = np.copy(X), np.copy(Y), np.copy(Z), copy.deepcopy(C)
					a = A[dim1][dim2][dim3]
					r = C.ttm(np.array([X[dim1]]), 0)\
						 .ttm(np.array([Y[dim2]]), 1)\
						 .ttm(np.array([Z[dim3]]), 2)[0][0][0]
					Xi, Yi, Zi = np.array([X[dim1]]), np.array([Y[dim2]]), np.array([Z[dim3]])
					X[dim1] = Xi-alpha*(r-a)*C.ttm(Yi, 1).ttm(Zi, 2).reshape((1,K))-alpha*beta*Xi
					Y[dim2] = Yi-alpha*(r-a)*C.ttm(Xi, 0).ttm(Zi, 2).reshape((1,K))-alpha*beta*Yi
					Z[dim3] = Zi-alpha*(r-a)*C.ttm(Xi, 0).ttm(Yi, 1).reshape((1,K))-alpha*beta*Zi
					C = C-alpha*(r-a)*np.kron(np.kron(Xi,Yi),Zi).reshape((K, K, K))-alpha*beta*C
					R = C.ttm(X, 0).ttm(Y, 1).ttm(Z, 2)
					loss_curr = compute_loss(A,R)
					if loss_curr > loss_last:
						X, Y, Z, C = _X, _Y, _Z, _C
						loss_curr = loss_last
			print "loss_curr", loss_curr
		if loss_curr < epsilon:
			return C, [X, Y, Z]
	return C, [X, Y, Z]


if __name__ == "__main__":
	# Input
	A = np.zeros((3, 4, 2))
	A[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
	A[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
	C, [X, Y, Z] = Gradient_Tensor_Decomposition(A, 2)
	R = C.ttm(X, 0).ttm(Y, 1).ttm(Z, 2)
	print R

