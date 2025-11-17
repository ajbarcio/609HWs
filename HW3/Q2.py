import numpy as np

np.set_printoptions(precision=8, suppress=True) 

def LUD(A):
    D = np.array(np.diag(np.diag(A)), dtype=float)
    # print(D)
    U = np.zeros_like(A, dtype=float)
    for i in range(len(A)):
        U += np.diag(np.diag(A, k=i), k=i)
    U=U-D
    # print(U)
    L = np.zeros_like(A, dtype=float)
    for i in range(len(A)):
        L += np.diag(np.diag(A, k=-i), k=-i)
    L=L-D
    # print(L)
    
    return L, U, D

A = np.array([[3,1,0],[1,3,1],[0,1,3]])
b = np.array([4,5,4], dtype = float)

if __name__=="__main__":

    print(np.linalg.eig(A)[0])

    L, U, D = LUD(A)

    D_inv = np.zeros_like(D, dtype=float)
    for i in range(len(A)):
        D_inv[i,i] = 1.0/float(D[i,i])

    # print(D_inv)
    B_J = -D_inv.dot(L+U)
    # print(B_J)
    B_J_inf = np.linalg.norm(B_J, np.inf)
    print("infinity norm", B_J_inf)
    rho_B_J = np.max(np.linalg.eig(B_J)[0])
    print("spectral radius:", rho_B_J)

    B_GS = -np.linalg.inv(L+D) @ U
    # print(B_GS)
    B_GS_inf = np.linalg.norm(B_GS, np.inf)
    print("infinity norm", B_GS_inf)
    rho_B_GS = np.max(np.linalg.eig(B_GS)[0])
    print("spectral radius:", rho_B_GS)

    omega_star = 2/(1+np.sqrt(1-rho_B_J**2))
    print("optimal SOR parameter", omega_star)
    print("for A sym. pos. def, omega should be between 0 and 2, and it is")


    B_SOR = np.linalg.inv(omega_star*L + D) @ ((1-omega_star)*D - omega_star*U)
    c_SOR = omega_star*np.linalg.inv(omega_star*L + D) @ b

    x =   np.array([1,1,1])
    x_0 = np.array([0,0,0])

    def Jacobi(iters, x_0):
        iters = iters+1
        out = np.zeros([iters, 6])
        out[0,0] = 0
        out[0,1:4] = x_0
        out[0,4]   = np.linalg.norm(x-x_0, np.inf)
        out[0,5]   = np.linalg.norm(x-x_0, np.inf)
        # print(out[0,:])
        x_i = x_0
        e_0 = np.linalg.norm(x-x_0, np.inf)
        for i in range(iters):
            out[i,0] = i
            e_i = np.linalg.norm(x_i-x, np.inf)
            out[i, 4] = e_i
            out[i, 5] = e_i/e_0
            if not i==iters-1:
                x_next = (B_J @ x_i + D_inv @ b).flatten()
                out[i+1,1:4] = x_next
                x_i = x_next
            e_0 = e_i
        return out

    def GaussSeidel(iters, x_0):
        iters = iters+1
        out = np.zeros([iters, 6])
        out[0,0] = 0
        out[0,1:4] = x_0
        out[0,4]   = np.linalg.norm(x-x_0, np.inf)
        out[0,5]   = np.linalg.norm(x-x_0, np.inf)
        # print(out[0,:])
        x_i = x_0
        e_0 = np.linalg.norm(x-x_0, np.inf)
        for i in range(iters):
            out[i,0] = i
            e_i = np.linalg.norm(x_i-x, np.inf)
            out[i, 4] = e_i
            out[i, 5] = e_i/e_0
            if not i==iters-1:
                x_next = (B_GS @ x_i + np.linalg.inv(L+D) @ b).flatten()
                out[i+1,1:4] = x_next
                x_i = x_next
            e_0 = e_i
        return out

    def SOR(iters, x_0):
        iters = iters+1
        out = np.zeros([iters, 6])
        out[0,0] = 0
        out[0,1:4] = x_0
        out[0,4]   = np.linalg.norm(x-x_0, np.inf)
        out[0,5]   = np.linalg.norm(x-x_0, np.inf)
        # print(out[0,:])
        x_prev = x_0
        x_i = np.zeros_like(x_0, dtype=float)
        e_0 = np.linalg.norm(x-x_0, np.inf)
        for i in range(iters):
            out[i,0] = i
            e_i = np.linalg.norm(x_i-x, np.inf)
            out[i, 4] = e_i
            out[i, 5] = e_i/e_0
            if not i==iters-1:
                x_next = (B_SOR @ x_i + c_SOR).flatten()
                out[i+1,1:4] = x_next
                x_i = x_next
            e_0 = e_i
        return out


    J_table = Jacobi(10, x_0)
    print(J_table)
    GS_table = GaussSeidel(10, x_0)
    print(GS_table)
    SOR_table = SOR(10, x_0)
    print(SOR_table)

    from array_to_latex import to_ltx

    to_ltx(J_table, frmt="{:.5f}", arraytype="tabular")
    to_ltx(GS_table, frmt="{:.5f}", arraytype="tabular")
    to_ltx(SOR_table, frmt="{:.5f}", arraytype="tabular")
    # print(J_table_formatted)