import operator, warnings
import tinyarray as ta
from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import sys

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

            # Creation from list
            l = a.tolist()
            b = ta.array(l)
            b_shape = shape_of_seq(l)

            # a_shape and b_shape are not always equal.
            # Example: a_shape == (0, 0), b_shape = (0,).

            assert isinstance(repr(b), str)
            assert_equal(b.ndim, len(b_shape))
            assert_equal(tuple(b.shape), b_shape)
            assert_equal(b.size, a.size)
            if a_shape != ():
                assert_equal(len(b), len(a))
                assert_equal(np.array(ta.array(b)), np.array(l))
            else:
                assert_equal(b.dtype, dtype)
                assert_raises(TypeError, len, b)
            if sys.version_info[:2] > (2, 6):
                # Python 2.6 does not have memoryview.
                assert_equal(memoryview(b).tobytes(), memoryview(a).tobytes())
            assert_equal(np.array(b), np.array(l))
            assert_equal(ta.transpose(l), np.transpose(l))

            # Here, the tinyarray is created via the buffer interface.  It's
            # possible to distinguish shape 0 from (0, 0).
            b = ta.array(a)

            assert isinstance(repr(b), str)
            assert_equal(b.ndim, len(b.shape))
            assert_equal(b.shape, a.shape)
            assert_equal(b.size, a.size)
            assert_equal(b, a)
            assert_equal(np.array(b), a)
            if a_shape != ():
                assert_equal(len(b), len(a))
            else:
                assert_raises(TypeError, len, b)
            if sys.version_info[:2] > (2, 6):
                # Python 2.6 does not have memoryview.
                assert_equal(memoryview(b).tobytes(), memoryview(a).tobytes())
            assert_equal(ta.transpose(b), np.transpose(a))

            # Check creation from NumPy matrix.  This only works for Python >
            # 2.6.  I don't know whether this is our bug or their's.
            if sys.version_info[:2] > (2, 6):
                if not isinstance(a_shape, tuple) or len(a_shape) <= 2:
                    b = ta.array(np.matrix(a))
                    assert_equal(b.ndim, 2)
                    assert_equal(b, np.matrix(a))

        l = []
        for i in range(16):
            l = [l]
        assert_raises(ValueError, ta.array, l, dtype)

        assert_raises(TypeError, ta.array, [0, [0, 0]], dtype)
        assert_raises(ValueError, ta.array, [[0], [0, 0]], dtype)
        assert_raises(ValueError, ta.array, [[0, 0], 0], dtype)
        assert_raises(ValueError, ta.array, [[0, 0], [0]], dtype)
        assert_raises(ValueError, ta.array, [[0, 0], [[0], [0]]], dtype)


def test_matrix():
    for l in [(), 3, (3,), ((3,)), (1, 2), ((1, 2), (3, 4))]:
        a = ta.matrix(l)
        b = np.matrix(l)
        assert_equal(a, b)
        a = ta.matrix(ta.array(l))
        assert_equal(a, b)

        if sys.version_info[:2] > (2, 6):
            # Creation of tinyarrays from NumPy matrices only works for Python >
            # 2.6.  I don't know whether this is our bug or their's.
            a = ta.matrix(b)
            assert_equal(a, b)

    for l in [(((),),), ((3,), ()), ((1, 2), (3,))]:
        assert_raises(ValueError, ta.matrix, l)


def test_conversion():
    for src_dtype in dtypes:
        for dest_dtype in dtypes:
            src = ta.zeros(3, src_dtype)
            tsrc = tuple(src)
            npsrc = np.array(tsrc)
            impossible = src_dtype is complex and dest_dtype in [int, float]
            for s in [src, tsrc, npsrc]:
                if impossible:
                    assert_raises(TypeError, ta.array, s, dest_dtype)
                else:
                    dest = ta.array(s, dest_dtype)
                    assert isinstance(dest[0], dest_dtype)
                    assert_equal(src, dest)

    # Check for overflow.
    long_overflow = [1e300, np.array([1e300])]
    # This check only works for Python 2.
    if 18446744073709551615 > sys.maxint:
        long_overflow.extend([np.array([18446744073709551615], np.uint64),
                              18446744073709551615])

    for s in long_overflow:
        assert_raises(OverflowError, ta.array, s, long)


def test_special_constructors():
    for dtype in dtypes:
        for shape in some_shapes:
            assert_equal(ta.zeros(shape, dtype), np.zeros(shape, dtype))
            assert_equal(ta.ones(shape, dtype), np.ones(shape, dtype))
        for n in [0, 1, 2, 3, 17]:
            assert_equal(ta.identity(n, dtype), np.identity(n, dtype))


def test_dot():
    # Check acceptance of non-tinyarray arguments.
    assert_equal(ta.dot([1, 2], (3, 4)), 11)

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
            assert_almost_equal(ta.dot(ta.array(a), ta.array(b)), np.dot(a, b),
                                13)

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
            assert_raises(ValueError, ta.dot, ta.array(a.tolist()),
                          ta.array(b.tolist()))
            assert_raises(ValueError, ta.dot, ta.array(a), ta.array(b))


def test_iteration():
    for dtype in dtypes:
        assert_raises(TypeError, tuple, ta.array(1, dtype))
        for shape in [0, 1, 2, 3, (1, 0), (2, 2), (17, 17),
                      (1, 1, 1), (2, 2, 1), (2, 0, 3)]:
            a = make(shape, dtype)
            assert_equal(tuple(a), a)


def test_as_dict_key():
    n = 100
    d = {}
    for dtype in dtypes + dtypes:
        for i in xrange(n):
            d[ta.array(xrange(i), dtype)] = i
        assert_equal(len(d), n)
    for i in xrange(n):
        assert_equal(d[tuple(xrange(i))], i)


def test_hash_equality():
    for thing in [float('nan'), float('inf'), float('-inf'),
                  0, -1, -1.0, -1 + 0j,
                  303, -312424, 2**31 - 1, 2**31, 2**53, 2**200,
                  -0.3, 1.7, 0.4j, -12.3j, 1 - 12.3j, 1.3 - 12.3j,
                  (), (-1,), (2,),
                  (0, 0), (-1, -1), (-5, 7), (3, -1, 0),
                  ((0, 0), (0, 0)), (((-1,),),)]:
        arr = ta.array(thing)
        if thing == thing:
            assert arr == thing
            assert not (arr != thing)
        assert_equal(hash(arr), hash(thing))


def test_broadcasting():
    for sa in [(), 1, (1, 1, 1, 1), 2, (3, 2), (4, 3, 2), (5, 4, 3, 2)]:
        for sb in [(), 1, (1, 1), (4, 1, 1), 2, (1, 2), (3, 1), (1, 3, 2)]:
            a = make(sa, int)
            b = make(sb, int)
            assert_equal(ta.array(a.tolist()) + ta.array(b.tolist()), a + b)
            assert_equal(ta.array(a) + ta.array(b), a + b)


def test_promotion():
    for dtypea in dtypes:
        for dtypeb in dtypes:
            a = make(3, dtypea)
            b = make(3, dtypeb)
            assert_equal(ta.array(a.tolist()) + ta.array(b.tolist()), a + b)
            assert_equal(ta.array(a) + ta.array(b), a + b)


def test_binary_operators():
    ops = operator
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for op in [ops.add, ops.sub, ops.mul, ops.div, ops.mod, ops.floordiv]:
            for dtype in dtypes:
                for shape in [(), 1, 3, (3, 2)]:
                    if dtype is complex and op in [ops.mod, ops.floordiv]:
                        continue
                    a = make(shape, dtype)
                    b = make(shape, dtype)
                    assert_equal(
                        op(ta.array(a.tolist()), ta.array(b.tolist())),
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
                assert_equal(op(ta.array(a.tolist())), op(a))
                assert_equal(op(ta.array(a)), op(a))


def test_unary_ufuncs():
    for name in ["negative", "abs", "absolute", "round", "floor", "ceil",
                 "conjugate"]:
        np_func = np.__dict__[name]
        ta_func = ta.__dict__[name]
        for dtype in dtypes:
            for shape in [(), 1, 3, (3, 2)]:
                a = make(shape, dtype)
                if dtype is complex and name in ["round", "floor", "ceil"]:
                    assert_raises(TypeError, ta_func, a.tolist())
                else:
                    assert_equal(ta_func(a.tolist()), np_func(a))
        for x in [-987654322.5, -987654321.5, -4.51, -3.51, -2.5, -2.0,
                   -1.7, -1.5, -0.5, -0.3, -0.0, 0.0, 0.3, 0.5, 1.5, 1.7,
                   2.0, 2.5, 3.51, 4.51, 987654321.5, 987654322.5]:
            if x == -0.5 and name == "round":
                # Work around an inconsistency in NumPy: on Unix, np.round(-0.5)
                # is -0.0, and on Windows it is 0.0, while np.ceil(-0.5) is -0.0
                # always.
                assert_equal(ta.round(-0.5), -0.0)
            else:
                assert_equal(ta_func(x), np_func(x))


def test_pickle():
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    for dtype in dtypes:
        for shape in some_shapes:
            a = ta.array(make(shape, dtype))
            assert_equal(pickle.loads(pickle.dumps(a)), a)


if __name__=='__main__':
    import nose
    nose.runmodule()
