import operator, warnings
import tinyarray as ta
from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

dtypes = [int, float, complex]


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
        for a_shape in [(), 0, 1, 2, 3,
                        (0, 0), (1, 0), (0, 1), (2, 2), (17, 17),
                        (0, 0, 0), (1, 1, 1), (2, 2, 1), (2, 0, 3)]:
            a = make(a_shape, dtype)

            l = a.tolist()
            b = ta.array(l)
            b_shape = shape_of_seq(l)

            # a_shape and b_shape are not always equal.  Example: a_shape ==
            # (0, 0), b_shape = (0,).

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
            assert_equal(memoryview(b).tobytes(), memoryview(a).tobytes())
            assert_equal(np.array(b), np.array(l))

            # Here, the tinyarray is created via the buffer interface.  It's
            # possible to distinguish shape 0 from (0, 0).
            # b = ta.array(a)
            # assert_equal(np.array(b), a)

        l = []
        for i in range(16):
            l = [l]
        assert_raises(ValueError, ta.array, l, dtype)

        assert_raises(TypeError, ta.array, [0, [0, 0]], dtype)
        assert_raises(ValueError, ta.array, [[0], [0, 0]], dtype)
        assert_raises(ValueError, ta.array, [[0, 0], 0], dtype)
        assert_raises(ValueError, ta.array, [[0, 0], [0]], dtype)
        assert_raises(ValueError, ta.array, [[0, 0], [[0], [0]]], dtype)


def test_special_constructors():
    for dtype in dtypes:
        for n in [0, 1, 2, 3, 17]:
            assert_equal(ta.zeros(n, dtype), np.zeros(n, dtype))
            assert_equal(ta.ones(n, dtype), np.ones(n, dtype))
            assert_equal(ta.identity(n, dtype), np.identity(n, dtype))


def test_dot():
    # Check acceptance of non-tinyarray arguments.
    assert_equal(ta.dot([1, 2], (3, 4)), 11)

    for dtype in dtypes:
        # The commented testcases can be added once there is support for
        # creating tinyarrays via the buffer interface.
        shape_pairs = [(1, 1), (2, 2), (3, 3),
                       # (0, 0),
                       # (0, (0, 1)), ((0, 1), 1),
                       # (0, (0, 2)), ((0, 2), 2),
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
        # (This is probably due to usage of SSE or parallelization.)
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

        # The commented out testcases do not work due to a bug in numpy:
        # PySequence_Check should return 0 for 0-d arrays.
        shape_pairs = [#((), 2), (2, ()),
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
            assert_raises(ValueError, ta.dot, ta.array(a), ta.array(b))


def test_iteration():
    for dtype in dtypes:
        assert_raises(TypeError, tuple, ta.array(1, dtype))
        for i in [0, 1, 2, 3, 15]:
            t = tuple(xrange(i))
            assert_equal(tuple(ta.array(t, dtype)), t)

            if i != 0:
                t = ta.identity(i, dtype)
                assert_equal(np.array(ta.array(tuple(t))), np.array(t))


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
    for tup in [0, -1, -1.0, -1 + 0j, -0.3, 1.7, 0.4j,
                -12.3j, 1 - 12.3j, 1.3 - 12.3j,
                (), (-1,), (2,),
                (0, 0), (-1, -1), (-5, 7), (3, -1, 0),
                ((0, 0), (0, 0)), (((-1,),),)]:
        arr = ta.array(tup)
        assert arr == tup
        assert not (arr != tup)
        assert hash(arr) == hash(tup)


def test_broadcasting():
    for sa in [(), 1, (1, 1, 1, 1), 2, (3, 2), (4, 3, 2), (5, 4, 3, 2)]:
        for sb in [(), 1, (1, 1), (4, 1, 1), 2, (1, 2), (3, 1), (1, 3, 2)]:
            a = make(sa, int)
            b = make(sb, int)
            assert_equal(ta.array(a.tolist()) + ta.array(b.tolist()), a + b)


def test_promotion():
    for dtypea in dtypes:
        for dtypeb in dtypes:
            a = make(3, dtypea)
            b = make(3, dtypeb)
            assert_equal(ta.array(a.tolist()) + ta.array(b.tolist()), a + b)


def test_binary_operators():
    ops = operator
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for op in [ops.add, ops.sub, ops.mul, ops.div, ops.mod, ops.floordiv]:
            for dtype in dtypes:
                for shape in [(), 1, 3]:
                    if dtype is complex and op in [ops.mod, ops.floordiv]:
                        continue
                    a = make(shape, dtype)
                    b = make(shape, dtype)
                    assert_equal(
                        op(ta.array(a.tolist()), ta.array(b.tolist())),
                        op(a, b))
