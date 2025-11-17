# import sys
# import os
# # Add parent directory to sys.path so HW4, HW5, HW3 are importable
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

import numpy as np
from fractions import Fraction
# from HW4.Q4 import solve, GaussSeidelStep

A = np.array([[1 , 1 ,      1 , 1 , 1],
              [-2 , -1 ,    0 , 1 , 2],
              [2 , 1/2 ,    0 , 1/2 , 2],
              [-4/3 , -1/6, 0 , 1/6 , 4/3],
              [2/3 , 1/24 , 0 , 1/24 , 2/3]])
b = np.array([0,0,1,0,0])

sol = np.linalg.solve(A, b)
fracA_str = np.vectorize(lambda x: Fraction(x).limit_denominator())(sol)
print(fracA_str)
print(A @ sol)
print(np.array([-32, -1, 0, 1, 32]).dot(sol))
print(np.array([64, 1, 0, 1, 64]).dot(sol))