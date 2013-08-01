import tinyarray
import numpy
from time import time

################################################################
# Mock a numpy like "module" using Python tuples.

def tuple_array(seq):
    return tuple(seq)

def tuple_zeros(shape, dtype):
    return (dtype(0),) * shape

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

def zeros(module, dtype, n=100000):
    zeros = module.zeros
    return list(zeros(2, dtype) for i in xrange(n))

def make_from_list(module, dtype, n=100000):
    array = module.array
    l = [dtype(e) for e in range(3)]
    return list(array(l) for i in xrange(n))

def dot(module, dtype, n=1000000):
    dot = module.dot
    a = module.zeros(2, dtype)
    b = module.zeros(2, dtype)
    for i in xrange(n):
        c = dot(a, b)

def dot_tuple(module, dtype, n=100000):
    dot = module.dot
    a = module.zeros(2, dtype)
    b = (dtype(0),) * 2
    for i in xrange(n):
        c = dot(a, b)

def compare(function, modules):
    print '{0}:'.format(function.__name__)
    for module in modules:
        # Execute the function once to make the following timings more
        # accurate.
        function(module, int)
        print "  {0:15}".format(module.__name__),
        for dtype in (int, float, complex):
            t = time()
            try:
                function(module, dtype)
            except:
                print " failed  ",
            else:
                print ' {0:.4f} s'.format(time() - t),
        print

def main():
    print '                   int       float     complex'
    modules = [tuples, tinyarray, numpy]
    compare(zeros, modules)
    compare(make_from_list, modules)
    compare(dot, modules)
    compare(dot_tuple, modules)

if __name__ == '__main__':
    main()
