import numpy as np

def G(x):
    return np.array([(x[0]**2+x[1]**2+8), x[0]*x[1]**2+x[0]+8])*1/10.0

def J(x):
    return np.array([[2/10*x[0], 2/10*x[1]],[(x[1]**2+1)/10, 2*x[0]*x[1]/10]])

def FPIM_step(x):
    return G(x)

def FPIM_solve(FPFun, initialGuess, stopVal=1e-16):
    x = initialGuess
    eval = FPFun(initialGuess)
    err = (np.linalg.norm(x-eval))
    i = 0
    while err>stopVal:
        eval = FPFun(x)
        err = (np.linalg.norm(x-eval))
        x = eval
        i+=1
        print(f"{i} iterations, err {err}", end="\r")
    print(f"{i} iterations, err {err}")
    return eval

def main():
    print(f"G([0,0]) = {G([0,0])}")
    print(f"G([0,1.5]) = {G([0,1.5])}")
    print(f"G([1.5,0]) = {G([1.5,0])}")
    print(f"G([1.5,1.5]) = {G([1.5,1.5])}")
    print(f"||J_G([0,0])||_\infty = {np.linalg.norm(J([0,0]))}")
    print(f"||J_G([0,1.5])||_\infty = {np.linalg.norm(J([0,1.5]))}")
    print(f"||J_G([1.5,0])||_\infty = {np.linalg.norm(J([1.5,0]))}")
    print(f"||J_G([1.5,1.5])||_\infty = {np.linalg.norm(J([1.5,1.5]))}")
    sol = FPIM_solve(G, [0,0])
    print(sol)
if __name__ == "__main__":
    main()