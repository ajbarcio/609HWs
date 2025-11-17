import numpy as np

np.set_printoptions(precision=4, suppress=True) 

def JacobiStep(A, b, x):
    x_k1 = np.zeros_like(x,dtype=float)
    for i in range(len(A)):
        sum_term = 0
        for j in range(len(A)):
            if i == j:
                continue
            sum_term +=  A[i,j]*x[j]
        x_k1[i] = (b[i]-sum_term)/A[i,i]
    r = b - A @ x
    return x_k1, r

def GaussSeidelStep(A, b, x):
    for i in range(len(A)):
        sum_term = 0
        for j in range(len(A)):
            if i == j:
                continue
            sum_term +=  A[i,j]*x[j]
        x[i] = (b[i]-sum_term)/A[i,i]
    r = b - A @ x
    return x, r

def GD_step(A, b, x):
    r = b - A @ x
    # print(A, r)
    if np.linalg.norm(r) == 0:
        return x, r
    else:
        alpha = r.dot(r)/(r.dot(A @ r))
        x = x + alpha*r
        return x, r

def CG_step(A, b, x, d):
    r = b - A @ x
    z = A @ d
    alpha = r.dot(d)/(d.dot(z))
    x = x + alpha * d
    r_next = r - alpha * z
    beta = r_next.dot(z)/(d.dot(z))
    d_next = -r_next+beta * d
    return x, r_next, d_next

def solve(A, b, x0, strat, maxiter, conv):
    x = x0
    for i in range(maxiter):
        x, r = strat(A, b, x)
        if np.linalg.norm(r, np.inf)<conv:
            break
    iters = i
    res = np.linalg.norm(r, np.inf)
    return x, iters, res

def solveCG(A, b, x0, strat, maxiter, conv):
    x = x0
    r = b - A @ x0
    d = -r
    for i in range(len(b)):
        x, r, d = strat(A, b, x, d)
    iters = i + 1
    res = np.linalg.norm(r, np.inf)
    return x, iters, res

def create_problem(size):

    A = np.zeros([size,size])
    b = np.zeros(size)
    x0 = np.zeros(size)
    for i in range(size):
        for j in range(size):
            A[i,j] = 1/(1+i+j+2)
        b[i] = 1/3*(np.sum(np.array([A[i,j] for j in range(size)])))
    # for i in range(size):
    # print(A)
    # print(b)
    return A, b, x0
def main():
    A1, b1, x01 = create_problem(16)
    A2, b2, x02 = create_problem(32)
    # A2 = np.array()

    x1, iters, res = solve(A1, b1, x01, GD_step, 1000, 1e-5)
    print(x1, iters, res)
    x1, iters, res = solveCG(A1, b1, x01, CG_step, 1000, 1e-5)
    print(x1, iters, res)

    x2, iters, res = solve(A2, b2, x02, GD_step, 1000, 1e-5)
    print(x2, iters, res)
    x2, iters, res = solveCG(A2, b2, x02, CG_step, 1000, 1e-5)
    print(x2, iters, res)

    A = np.array([[10, 1,  2,    3,  4],
                [1,  9, -1,    2, -3],
                [2, -1,  7,    3, -5],
                [3,  2,  3,   12, -1],
                [4, -3, -5,   -1, 15]])
    b = np.array([12, -27, 14, -17, 12])

    x_J,  iters_J, res_J   = solve  (A, b, np.zeros(len(b)), JacobiStep, 1000, 1e-5)
    x_GS, iters_GS, res_GS = solve  (A, b, np.zeros(len(b)), GaussSeidelStep, 1000, 1e-5)
    x_GD, iters_GD, res_GD = solve  (A, b, np.zeros(len(b)), GD_step, 1000, 1e-5)
    x_CG, iters_CG, res_CG = solveCG(A, b, np.zeros(len(b)), CG_step, 1000, 1e-5)
    print("----------------------------------------------------")
    print("Jacobi      :", x_J, iters_J, res_J)
    print("Gauss Seidel:", x_GS, iters_GS, res_GS)
    print("Grad Descent:", x_GD, iters_GD, res_GD)
    print("Conjgte Grad:", x_CG, iters_CG, res_CG)

if __name__ == "__main__":
    main()