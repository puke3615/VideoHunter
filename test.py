import numpy as np
import random
import time

a = [
    [1, 2, 4],
    [2, 6, 5],
    [1, 2, 3],
]

a = np.array(a)
print np.max(a)
print np.argmax(a)
print np.amax(a)
names = ['A', 'B', 'C']


import utils
print utils.parse_predict(a, names, unique=True)
