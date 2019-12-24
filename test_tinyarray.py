# Copyright 2012-2016 Tinyarray authors.
#
# This file is part of Tinyarray.  It is subject to the license terms in the
# file LICENSE.rst found in the top-level directory of this distribution and
# at https://gitlab.kwant-project.org/kwant/tinyarray/blob/master/LICENSE.rst.
# A list of Tinyarray authors can be found in the README.rst file at the
# top-level directory of this distribution and at
# https://gitlab.kwant-project.org/kwant/tinyarray.

import operator, warnings
import platform
import itertools as it
import tinyarray as ta
from pytest import raises, xfail
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import sys
import random


# numpy.testing.assert_equal() is still used even with pytest, since it does
# the right thing for arrays containing anomalous values (e.g. NaNs).


def machine_wordsize():
    bits, _ = platform.architecture()
    if bits == '32bit':
        return 4
    elif bits == '64bit':
        return 8
    else:
        raise RuntimeError('unknown architecture', bits)

dtypes = [int, float, complex]

some_shapes = [(), 0, 1, 2, 3,
               (0, 0), (1, 0), (0, 1), (2, 2), (17, 17),
               (0, 0, 0), (1, 1, 1), (2, 2, 1), (2, 0, 3)]


def make(shape, dtype):
    result = np.arange(np.prod(shape), dtype=int)
    if dtype in (float, complex):
        result = result + 0.1 * result
    if dtype == complex:
        result = result + -0.5j * result
    return result.reshape(shape)


def shape_of_seq(seq, r=()):
    try:
        l = len(seq)
    except:
        return r
    if l == 0:
        return r + (0,)
    return shape_of_seq(seq[0], r + (l,))


def test_array():
    for dtype in dtypes:
        for a_shape in some_shapes:
            a = make(a_shape, dtype)

            # Creation from list.  This also tests creation from scalars.
            l = a.tolist()
            b = ta.array(l)
            b_shape = shape_of_seq(l)

            # a_shape and b_shape are not always equal.
            # Example: a_shape == (0, 0), b_shape = (0,).

            assert isinstance(repr(b), str)
            assert b.ndim == len(b_shape)
            assert tuple(b.shape) == b_shape
            assert b.size == a.size
            if a_shape != ():
                assert len(b) == len(a)
                assert_equal(np.array(ta.array(b)), np.array(l))
            else:
                assert b.dtype == dtype
                raises(TypeError, len, b)
            if sys.version_info[:2] > (2, 6):
                # Python 2.6 does not have memoryview.
                assert memoryview(b).tobytes() == memoryview(a).tobytes()
            assert_equal(np.array(b), np.array(l))
            assert ta.transpose(l) == np.transpose(l)

            # Here, the tinyarray is created via the buffer interface.  It's
            # possible to distinguish shape 0 from (0, 0).
            b = ta.array(a)

            # This tests creation of arrays from non-C-contiguous buffers.
            assert b == ta.array(a.transpose()).transpose()

            assert isinstance(repr(b), str)
            assert b.ndim == len(b.shape)
            assert b.shape == a.shape
            assert b.size == a.size
            assert b == a
            assert_equal(np.array(b), a)
            if a_shape != ():
                assert len(b) == len(a)
            else:
                raises(TypeError, len, b)
            if sys.version_info[:2] > (2, 6):
                # Python 2.6 does not have memoryview.
                assert memoryview(b).tobytes() == memoryview(a).tobytes()
            assert ta.transpose(b) == np.transpose(a)

            # Check creation from NumPy matrix.  This only works for Python >
            # 2.6.  I don't know whether this is our bug or their's.
            if sys.version_info[:2] > (2, 6):
                if not isinstance(a_shape, tuple) or len(a_shape) <= 2:
                    b = ta.array(np.matrix(a))
                    assert b.ndim == 2
                    assert b == np.matrix(a)

        l = []
        for i in range(16):
            l = [l]
        raises(ValueError, ta.array, l, dtype)

        raises(TypeError, ta.array, [0, [0, 0]], dtype)
        raises(ValueError, ta.array, [[0], [0, 0]], dtype)
        raises(ValueError, ta.array, [[0, 0], 0], dtype)
        raises(ValueError, ta.array, [[0, 0], [0]], dtype)
        raises(ValueError, ta.array, [[0, 0], [[0], [0]]], dtype)


def test_matrix():
    for l in [(), 3, (3,), ((3,)), (1, 2), ((1, 2), (3, 4))]:
        a = ta.matrix(l)
        b = np.matrix(l)
        assert a == b
        assert a.shape == b.shape
        a = ta.matrix(ta.array(l))
        assert a == b
        assert a.shape == b.shape
        a = ta.matrix(np.array(l))
        assert a == b
        assert a.shape == b.shape

        if sys.version_info[:2] > (2, 6):
            # Creation of tinyarrays from NumPy matrices only works for Python >
            # 2.6.  I don't know whether this is our bug or their's.
            a = ta.matrix(b)
            assert a == b

    for l in [(((),),), ((3,), ()), ((1, 2), (3,))]:
        raises(ValueError, ta.matrix, l)


def test_conversion():
    for src_dtype in dtypes:
        for dest_dtype in dtypes:
            src = ta.zeros(3, src_dtype)
            tsrc = tuple(src)
            npsrc = np.array(tsrc)
            impossible = src_dtype is complex and dest_dtype in [int, float]
            for s in [src, tsrc, npsrc]:
                if impossible:
                    raises(TypeError, ta.array, s, dest_dtype)
                else:
                    dest = ta.array(s, dest_dtype)
                    assert isinstance(dest[0], dest_dtype)
                    assert src == dest

    # Check for overflow.
    long_overflow = [1e300, np.array([1e300])]
    # This check only works for Python 2.
    if 18446744073709551615 > sys.maxsize:
        long_overflow.extend([np.array([18446744073709551615], np.uint64),
                              18446744073709551615])

    for s in long_overflow:
        raises(OverflowError, ta.array, s, int)


def test_special_constructors():
    for dtype in dtypes:
        for shape in some_shapes:
            assert ta.zeros(shape, dtype) == np.zeros(shape, dtype)
            assert ta.ones(shape, dtype) == np.ones(shape, dtype)
        for n in [0, 1, 2, 3, 17]:
            assert ta.identity(n, dtype) == np.identity(n, dtype)


def test_dot():
    # Check acceptance of non-tinyarray arguments.
    assert ta.dot([1, 2], (3, 4)) == 11

    for dtype in dtypes:
        shape_pairs = [(1, 1), (2, 2), (3, 3),
                       (0, 0),
                       (0, (0, 1)), ((0, 1), 1),
                       (0, (0, 2)), ((0, 2), 2),
                       (1, (1, 2)), ((2, 1), 1),
                       (2, (2, 1)), ((1, 2), 2),
                       (2, (2, 3)), ((3, 2), 2),
                       ((1, 1), (1, 1)), ((2, 2), (2, 2)),
                       ((3, 3), (3, 3)), ((2, 3), (3, 2)), ((2, 1), (1, 2)),
                       ((2, 3, 4), (4, 3)),
                       ((2, 3, 4), 4),
                       ((3, 4), (2, 4, 3)),
                       (4, (2, 4, 3))]

        # We have to use almost_equal here because the result of numpy's dot
        # does not always agree to the last bit with a naive implementation.
        # (This is probably due to their usage of SSE or parallelization.)
        #
        # On my machine in summer 2012 with Python 2.7 and 3.2 the program
        #
        # import numpy as np
        # a = np.array([13.2, 14.3, 15.4, 16.5])
        # b = np.array([-5.0, -3.9, -2.8, -1.7])
        # r = np.dot(a, b)
        # rr = sum(x * y for x, y in zip(a, b))
        # print(r - rr)
        #
        # outputs 2.84217094304e-14.
        for sa, sb in shape_pairs:
            a = make(sa, dtype)
            b = make(sb, dtype) - 5
            dta = ta.dot(ta.array(a), ta.array(b))
            dnp = np.dot(a, b)
            # This circumvents a build error on Numpy 1.12.0, where numpy's
            # iscomplexobj does not return True for complex tinyarrays.
            # In this case we do the test per element.
            if np.__version__ != '1.12.0':
                assert_almost_equal(dta, dnp, 13)
            elif (getattr(dta, "dtype", None) is complex and
                  getattr(dta, "shape", None) is not None and
                  len(dta) > 0):
                idx = it.product(*[range(i) for i in dta.shape])
                for i in idx:
                    assert_almost_equal(dta[i], dnp[i], 13)
            else:
                assert_almost_equal(dta, dnp, 13)

        shape_pairs = [((), 2), (2, ()),
                       (1, 2),
                       (1, (2, 2)), ((1, 1), 2),
                       ((2, 2), (3, 2)),
                       ((2, 3, 2), (4, 3)),
                       ((2, 3, 4), 3),
                       ((3, 3), (2, 4, 3)),
                       (3, (2, 4, 3))]
        for sa, sb in shape_pairs:
            a = make(sa, dtype)
            b = make(sb, dtype) - 5
            raises(ValueError, ta.dot, ta.array(a.tolist()),
                   ta.array(b.tolist()))
            raises(ValueError, ta.dot, ta.array(a), ta.array(b))


def test_iteration():
    for dtype in dtypes:
        raises(TypeError, tuple, ta.array(1, dtype))
        for shape in [0, 1, 2, 3, (1, 0), (2, 2), (17, 17),
                      (1, 1, 1), (2, 2, 1), (2, 0, 3)]:
            a = make(shape, dtype)
            assert tuple(ta.array(a)) == tuple(a)


def test_as_dict_key():
    # TODO: remove this once gitlab issue 16 is closed
    if sys.version_info >= (3, 8, 0):
        xfail('New version of tuple hash not supported yet.')

    n = 100
    d = {}
    for dtype in dtypes + dtypes:
        for i in range(n):
            d[ta.array(range(i), dtype)] = i
        assert len(d) == n
    for i in range(n):
        assert d[tuple(range(i))] == i


def test_hash_equality():
    # TODO: remove this once gitlab issue 16 is closed
    if sys.version_info >= (3, 8, 0):
        xfail('New version of tuple hash not supported yet.')

    random.seed(123)

    # These refer to the width of integers stored in a tinyarray.ndarray_int.
    int_bits = (8 * ta.dtype_size[int]) - 1  # 8 bits per byte, minus 1 sign bit
    maxint = 2**(int_bits)

    special = [float('nan'), float('inf'), float('-inf'),
               0, -1, -1.0, -1 + 0j,
               303, -312424, -0.3, 1.7, 0.4j, -12.3j, 1 - 12.3j, 1.3 - 12.3j,
               (), (-1,), (2,),
               (0, 0), (-1, -1), (-5, 7), (3, -1, 0),
               ((0, 1), (2, 3)), (((-1,),),)]
    powers = [sign * (2**e + a) for sign in [1, -1] for a in [-1, 0, 1]
              for e in range(int_bits)]
    powers.extend([2**int_bits - 1, -2**int_bits, -2**int_bits + 1])
    small_random_ints = (random.randrange(-2**16, 2**16) for i in range(1000))
    large_random_ints = (random.randrange(-maxint, maxint) for i in range(1000))
    small_random_floats = (random.gauss(0, 1) for i in range(1000))
    large_random_floats = (random.gauss(0, 1e100) for i in range(1000))

    for collection in [special, powers,
                       small_random_ints, large_random_ints,
                       small_random_floats, large_random_floats]:
        for thing in collection:
            arr = ta.array(thing)
            if thing == thing:
                assert arr == thing
                assert not (arr != thing)
            assert hash(arr) == hash(thing), repr(thing)


def test_broadcasting():
    for sa in [(), 1, (1, 1, 1, 1), 2, (3, 2), (4, 3, 2), (5, 4, 3, 2)]:
        for sb in [(), 1, (1, 1), (4, 1, 1), 2, (1, 2), (3, 1), (1, 3, 2)]:
            a = make(sa, int)
            b = make(sb, int)
            assert ta.array(a.tolist()) + ta.array(b.tolist()) == a + b
            assert ta.array(a) + ta.array(b) == a + b


def test_promotion():
    for dtypea in dtypes:
        for dtypeb in dtypes:
            a = make(3, dtypea)
            b = make(3, dtypeb)
            assert ta.array(a.tolist()) + ta.array(b.tolist()) == a + b
            assert ta.array(a) + ta.array(b) == a + b


def test_binary_operators():
    ops = operator
    operations = [ops.add, ops.sub, ops.mul, ops.mod, ops.floordiv, ops.truediv]
    if sys.version_info.major < 3:
        operations.append(ops.div)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for op in operations:
            for dtype in dtypes:
                for shape in [(), 1, 3, (3, 2)]:
                    if dtype is complex and op in [ops.mod, ops.floordiv]:
                        continue
                    a = make(shape, dtype)
                    b = make(shape, dtype)
                    assert_equal(op(ta.array(a.tolist()), ta.array(b.tolist())),
                                 op(a, b))
                    assert_equal(op(ta.array(a), ta.array(b)), op(a, b))



def test_binary_ufuncs():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for name in ["add", "subtract", "multiply", "divide",
                     "remainder", "floor_divide"]:
            np_func = np.__dict__[name]
            ta_func = ta.__dict__[name]
            for dtype in dtypes:
                for shape in [(), 1, 3, (3, 2)]:
                    if dtype is complex and \
                            name in ["remainder", "floor_divide"]:
                        continue
                    a = make(shape, dtype)
                    b = make(shape, dtype)
                    assert_equal(ta_func(a.tolist(), b.tolist()),
                                 np_func(a, b))
                    assert_equal(ta_func(a, b), np_func(a, b))


def test_unary_operators():
    ops = operator
    for op in [ops.neg, ops.pos, ops.abs]:
        for dtype in dtypes:
            for shape in [(), 1, 3, (3, 2)]:
                a = make(shape, dtype)
                assert op(ta.array(a.tolist())) == op(a)
                assert op(ta.array(a)) == op(a)


def test_unary_ufuncs():
    for name in ["negative", "abs", "absolute", "round", "floor", "ceil",
                 "conjugate"]:
        np_func = np.__dict__[name]
        ta_func = ta.__dict__[name]
        for dtype in dtypes:
            for shape in [(), 1, 3, (3, 2)]:
                a = make(shape, dtype)
                if dtype is complex and name in ["round", "floor", "ceil"]:
                    raises(TypeError, ta_func, a.tolist())
                else:
                    assert ta_func(a.tolist()) == np_func(a)
        for x in [-987654322.5, -987654321.5, -4.51, -3.51, -2.5, -2.0,
                   -1.7, -1.5, -0.5, -0.3, -0.0, 0.0, 0.3, 0.5, 1.5, 1.7,
                   2.0, 2.5, 3.51, 4.51, 987654321.5, 987654322.5]:
            if x == -0.5 and name == "round":
                # Work around an inconsistency in NumPy: on Unix, np.round(-0.5)
                # is -0.0, and on Windows it is 0.0, while np.ceil(-0.5) is -0.0
                # always.
                assert ta.round(-0.5) == -0.0
            else:
                assert ta_func(x) == np_func(x)


def test_other_scalar_types():
    types = [np.int16, np.int32, np.int64,
             np.float16, np.float32, np.float64]
    for t in types:
        a = t(123.456)
        assert_equal(ta.array(a), np.array(a))
        assert_equal(ta.matrix(a), np.matrix(a))


def test_sizeof():
    obj = object()
    word_size = machine_wordsize()
    for shape in some_shapes:
        for dtype in dtypes:
            a = ta.zeros(shape, dtype)
            sizeof = a.__sizeof__()
            # basic buffer size
            n_elements = a.size
            # if the array is > 1D then the shape is stored
            # at the start of the buffer
            if len(a.shape) > 1:
                n_elements += (a.ndim * machine_wordsize() +
                               ta.dtype_size[dtype] - 1) // ta.dtype_size[dtype]
            buffer_size = n_elements * ta.dtype_size[dtype]

            # A Basic Python object has 3 pointer-sized members, or 5 if in
            # debug mode.
            debug = hasattr(sys, "gettotalrefcount")
            sizeof_should_be = (buffer_size
                                + machine_wordsize() * (5 if debug else 3))
            assert sizeof == sizeof_should_be


def test_comparison():
    ops = operator
    for op in [ops.ge, ops.gt, ops.le, ops.lt, ops.eq, ops.ne]:
        for dtype in (int, float, complex):
            for left, right in it.product((np.zeros, np.ones), repeat=2):
                for shape in [(), (1,), (2,), (2, 2), (2, 2, 2), (2, 3)]:
                    a = left(shape, dtype)
                    b = right(shape, dtype)
                    if dtype is complex and op not in [ops.eq, ops.ne]:
                        # unorderable types
                        raises(TypeError, op, ta.array(a), ta.array(b))
                    else:
                        # passing the same object
                        same = ta.array(a)
                        assert op(same, same) == op(a.tolist(), a.tolist())
                        # passing different objects, but equal
                        assert (op(ta.array(a), ta.array(a)) ==
                                op(a.tolist(), a.tolist()))
                        # passing different objects, not equal
                        assert (op(ta.array(a), ta.array(b)) ==
                                op(a.tolist(), b.tolist()))
                # test different ndims and different shapes
                for shp1, shp2 in [((2,), (2, 2)), ((2, 2), (2, 3))]:
                    a = left(shp1, dtype)
                    b = right(shp2, dtype)
                    if op not in (ops.eq, ops.ne):
                        # unorderable types
                        raises(TypeError, op, ta.array(a), ta.array(b))


def test_pickle():
    import pickle

    for dtype in dtypes:
        for shape in some_shapes:
            a = ta.array(make(shape, dtype))
            assert pickle.loads(pickle.dumps(a)) == a
