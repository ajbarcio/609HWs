import numpy as np
from fractions import Fraction

A = np.array([[1,-5,1],[10,0,20],[5,0,-1]])
# P = np.array([[1,0,0],[0,0,1],[0,1,0]])

b = np.array([7,6,4])

print(np.linalg.solve(A,b))

# L = np.array([ ])