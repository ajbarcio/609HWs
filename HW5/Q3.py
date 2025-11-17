import sys
import os
# Add parent directory to sys.path so HW4, HW5, HW3 are importable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from HW4.Q4 import solve, solveCG, GD_step, CG_step

import numpy as np
from matplotlib import pyplot as plt

def phi(x):
    return np.e**(x**2)

def thomas(A,b):
    N = len(A)
    L = np.eye(len(A))
    d=b
    a = np.diagonal(A, offset=-1)
    a = np.insert(a, 0, 0)
    b = np.diagonal(A)
    c = np.diagonal(A, offset=1)
    c = np.append(c, 0)
    N = len(a)
    cp = np.zeros(N,dtype='float64') # store tranformed c or c'
    dp = np.zeros(N,dtype='float64') # store transformed d or d'
    X = np.zeros(N,dtype='float64') # store unknown coefficients
    
    # Perform Forward Sweep
    # Equation 1 indexed as 0 in python
    cp[0] = c[0]/b[0]  
    dp[0] = d[0]/b[0]
    # Equation 2, ..., N (indexed 1 - N-1 in Python)
    for i in np.arange(1,(N),1):
        dnum = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/dnum
        dp[i] = (d[i]-a[i]*dp[i-1])/dnum
    
    # Perform Back Substitution
    X[(N-1)] = dp[N-1]  # Obtain last xn 

    for i in np.arange((N-2),-1,-1):  # use x[i+1] to obtain x[i]
        X[i] = (dp[i]) - (cp[i])*(X[i+1])
    
    return(X)

def C(x):
    return 4*x**2+2

def f(x):
    return 0

def solve_BVP(C, strat, h, f, phi0, phi1):
    n = int(1/h)
    A = np.zeros((n-1,n-1))
    # print(len(A))
    # print(len(A)**2)
    for i in range(len(A)):
        A[i,i] = 2+C((i+1)*h)*h**2
        if not i==0:
            A[i,i-1] = -1
        if not i==n-2:
            A[i,i+1] = -1
    # print(A)
    b = np.ones(len(A))
    b *= h**2
    for i in range(len(b)):
        b[i] *= f((i+1)*h)
        if i==0:
            b[i]+=phi0
        if i==n-2:
            b[i]+=phi1
    # print(b)
    if strat=='GD':
        sol = solve(A, b, np.ones_like(b)*(phi0+phi1)/2, GD_step, 10000, 1e-8)[0]
    elif strat=='CG':
        sol = solveCG(A, b, np.ones_like(b)*(phi0+phi1)/2, CG_step, 10000, 1e-8)[0]
    elif strat=='thomas':
        sol = thomas(A,b)
    return sol

def main():
    strat = 'CG'
    print(strat)
    for i in [x+1 for x in range(14)]:
        h = 2**(-i)
        if i < 15:
            sol = solve_BVP(C, strat, h, f, 1, np.e)
            # print(sol[0])
            n = int(1/h)
            x = np.array([(i+1)*h for i in range(n-1)])
            # n = int(1/h)
            # print(n)
            # print(i)
            err = np.linalg.norm(sol - phi(x), ord=np.inf)

            print(f"{h:<12g}  " "&"
                f"{err:<18.12g}  " "&"
                f"{(err/h):<18.12g}  " "&"
                f"{(err/(h**2)):<18.12g}  " "&"
                f"{(err/(h**3)):<18.12g}" "\\\\")
            # print("--")
        # if i==14:
        #     plt.plot(x, sol)
        #     plt.plot(x, phi(x))
        #     plt.show()
if __name__ == "__main__":
    main()
