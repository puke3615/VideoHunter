import numpy as np

a = list(range(10))

print a[0::2]


b = np.zeros([10, 10])

c = b.copy()
# print b

c = c[0::2, 0::2]
print c
