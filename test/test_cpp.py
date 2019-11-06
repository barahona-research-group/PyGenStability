import pygenstability.pygenstability as pygen
import numpy as np

print('add', pygen.cpp.add([1,2],[4,2]))
print('multt', pygen.cpp.mult(np.array([1,2]), np.array([3,4])))
