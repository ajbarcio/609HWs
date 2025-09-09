import numpy as np

As = np.array([[[ 2,-1, 0, 0],
                [-1, 2,-1, 0],
                [ 0,-1, 2,-1],
                [ 0, 0,-1, 2]],
                
               [[1    ,0.5  ,1.0/3,1.0/4],
                [0.5  ,1.0/3,1.0/4,1.0/5],
                [1.0/3,1.0/4,1.0/5,1.0/6],
                [1.0/4,1.0/5,1.0/6,1.0/7]]])

for A in As:
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    L[0,0] = np.sqrt(A[0,0])
    for j in np.arange(1,np.shape(A)[0]):
        L[j,0] = A[j,0]/L[0,0]
    for i in np.arange(1,np.shape(A)[0]-1):
        L[i,i] = np.sqrt(A[i,i]-np.sum(np.array([L[i,k]**2 for k in range(i)])))
        for j in np.arange(i+1,np.shape(A)[0]):
            L[j,i] = 1/L[i,i]*(A[j,i]-np.sum(np.array([L[j,k]*L[i,k] for k in range(i)])))
    L[-1,-1] = np.sqrt(A[-1,-1]-np.sum(np.array([L[-1,k]**2 for k in range(np.shape(A)[0]-1)])))
    U = L.T

    print(A)
    # print('should be', '\n',np.linalg.cholesky(A))
    print(L)
    print(U)
    print(L.dot(U))
    # for i in np.arange(1,np.shape(A)[0]):
    #     for j in np.arange(1,np.shape(A)[0]):
