tinyarray implements a subset of NumPy's functionality in a way optimized for
small arrays, both in terms memory usage of runtime.  For small arrays,
speedups of 3 to 35 times are measured compared to NumPy.  Unlike NumPy arrays,
tinyarrays are immutable and thus can be used as dictionary keys.

To install, run

   python setup.py build
   python setup.py install
