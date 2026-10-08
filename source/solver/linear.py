import numpy


def solve(K, f):
    return numpy.linalg.solve(K.values, f)
