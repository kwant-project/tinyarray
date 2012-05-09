import tinyarray
import numpy
from time import time

################################################################
# Mock a numpy like "module" using Python tuples.

def tuple_array(seq, dtype):
    return tuple(seq)

def tuple_zeros(shape, dtype):
    return (0,) * shape

def tuple_dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

class Empty(object):
    pass

tuples = Empty()
tuples.__name__ = 'tuples'
tuples.array = tuple_array
tuples.zeros = tuple_zeros
tuples.dot = tuple_dot

################################################################

def zeros(module, n=100000):
    zeros = module.zeros
    return list(zeros(2, int) for i in xrange(n))

def make_from_list(module, n=100000):
    array = module.array
    l = range(3)
    return list(array(l, int) for i in xrange(n))

def dot(module, n=1000000):
    dot = module.dot
    a = module.zeros(2, int)
    b = module.zeros(2, int)
    for i in xrange(n):
        c = dot(a, b)

def dot_tuple(module, n=100000):
    dot = module.dot
    a = module.zeros(2, int)
    b = (0, 0)
    for i in xrange(n):
        c = dot(a, b)

def compare(function, modules):
    print '****', function.__name__, '****'
    for module in modules:
        t = time()
        try:
            function(module)
        except:
            print "{0:15} failed".format(module.__name__)
        else:
            print '{0:15} {1:.4f} s'.format(module.__name__, time() - t)

def main():
    modules = [tuples, tinyarray, numpy]
    compare(zeros, modules)
    compare(make_from_list, modules)
    compare(dot, modules)
    compare(dot_tuple, modules)

if __name__ == '__main__':
    main()
