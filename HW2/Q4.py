import numpy as np
from Q2 import LUD

np.set_printoptions(precision=5, suppress=True) 

threshold = 1e-2

B = np.array(4*np.eye(3)+
             np.diag(2*[-1], k=1)+
             np.diag(2*[-1], k=-1), dtype=float)
# print(B)
A = np.array(np.block([[B,                -np.eye(3), np.zeros_like(B)],
              [-np.eye(3),       B,          -np.eye(3)],
              [np.zeros_like(B), -np.eye(3), B]]), dtype=float)
# print(A)
b = (0,0,1,0,0,1,0,0,1)
L, U, D = LUD(A)
# print(L)
# print(U)
# print(D)

def JacobiStep(A, b, x):
    x_k1 = np.zeros_like(x,dtype=float)
    for i in range(len(A)):
        sum_term = 0
        for j in range(len(A)):
            if i == j:
                continue
            sum_term +=  A[i,j]*x[j]
        x_k1[i] = (b[i]-sum_term)/A[i,i]
    return x_k1

def GaussSeidelStep(A, b, x):
    for i in range(len(A)):
        sum_term = 0
        for j in range(len(A)):
            if i == j:
                continue
            sum_term +=  A[i,j]*x[j]
        x[i] = (b[i]-sum_term)/A[i,i]
    return x

i = 0
x_0 = np.zeros(len(A))
res_norm = np.linalg.norm(A@x_0 - b)
res_norm_prev = res_norm
x_jacobi=x_0
x_gaussS=x_0
# print("")
jacobi_out = []
gaussS_out = []
while res_norm > threshold:
    if i < 100:
        x_jacobi = JacobiStep(A, b, x_jacobi)
        # print(i, end='\r')
        res_norm = np.linalg.norm(A@x_jacobi - b)
        jacobi_out.append(np.hstack([i, res_norm, res_norm/res_norm_prev]))
        res_norm_prev = res_norm
    i+=1
print(x_jacobi)
print(np.array(jacobi_out))

i = 0
res_norm = np.linalg.norm(A@x_0 - b)
res_norm_prev = res_norm
while res_norm > threshold:
    if i < 100:
        x_gaussS = GaussSeidelStep(A, b, x_gaussS)
        # print(i, end='\r')
        res_norm = np.linalg.norm(A@x_gaussS - b)
        gaussS_out.append(np.hstack([i, res_norm, res_norm/res_norm_prev]))
        res_norm_prev = res_norm
    i+=1
print(x_gaussS)
print(np.array(gaussS_out))

# sol = np.linalg.solve(A, b)
# print(sol) # just checking teehee