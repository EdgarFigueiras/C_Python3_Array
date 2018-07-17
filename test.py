import sys
import cArray
import numpy as np
from numpy import median
import random

y = np.random.random((100, 100, 100))
z = np.zeros((100, 3))

print(cArray.version())

#print(y)

print("---")

#print(z)

ranges=np.zeros((3,2))

ranges[0][0]=2
ranges[0][1]=5
ranges[1][0]=1
ranges[1][1]=4
ranges[2][0]=2
ranges[2][1]=8

cArray.matrix3DprobRange(y,z,ranges)

#print(y)

print("---")

print(z)

#media = median(z)

#print(media)
