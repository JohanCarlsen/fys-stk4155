import numpy as np

a = np.arange(10)
b = np.zeros_like(a, dtype=bool)
shuf = np.random.choice(a, replace=False, size=5)
ind = np.array_split(shuf, 3)
b[ind[0]] = 1
print(a[b])
print(a[~b])