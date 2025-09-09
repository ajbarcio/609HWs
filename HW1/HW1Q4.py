import numpy as np
from matplotlib import pyplot as plt


fig, axs = plt.subplots(2,4)
ns = [1,2,3,4,5,6,7,8]
for n in ns:
    print('n is', n, '-----------------------------------------')
    N = 2**n - 1

    eps = 10**-3
    h = 2**(-n)

    b = [-eps/h**2]*(N-1)
    c = b
    a = [(2*eps+h**2)/h**2]*(N)
    f = [2*(i+1)*h+1 for i in range(N)]

    print('a', a)
    print('b', b)
    print('f', f)

    omega = []
    l = []
    for i in range(N):
        if i == 0:
            # print('first')
            omega.append(a[0])
        elif i == N-1:
            # print('last')
            l.append(c[i-1]/omega[i-1])
            omega.append(a[-1])
        else:
            # print('middle')
            l.append(c[i-1]/omega[i-1])
            omega.append(a[i]-l[i-1]*b[i-1])
    print('l, omega', l, omega)

    y=[]
    y.append(f[0])
    i = 1
    while i < (N):
        # print(i)
        y.append(f[i]-l[i-1]*y[i-1])
        i+=1
    print('y', y)

    u=[0]*N
    i = N-2
    u[-1] = y[-1]/omega[-1]
    while i >= 0:
        u[i] = (y[i]-b[i]*u[i+1])/omega[i]
        i-=1
    print('u', u)   

    x =  [h*(i+1) for i in range(len(u))]
    plt.subplot(2,4,ns.index(n)+1)
    plt.plot(x, u)
    plt.scatter(x,u,color='red')
    plt.title(f'n={n}')
    plt.xlabel(f'x')
    plt.ylabel(f'u')
plt.tight_layout()
plt.show()
    
# print(np.array())

