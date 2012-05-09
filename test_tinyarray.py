import tinyarray as ta
from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_equal


def test_array():
    for shape in [(), 0, 1, 2, 3, (0, 0), (1, 0), (0, 1), (2, 2), (17, 17),
                  (0, 0, 0), (1, 1, 1), (2, 2, 1), (2, 0, 3)]:
        a = np.arange(np.prod(shape), dtype=int).reshape(shape)

        l = a.tolist()
        b = ta.array(l)
        assert isinstance(repr(b), str)
        if shape != ():
            assert_equal(len(b), len(a))
        else:
            assert_raises(TypeError, len, b)
        assert_equal(memoryview(b).tobytes(), memoryview(a).tobytes())
        assert_equal(np.array(b), np.array(l))

        # Here, the tinyarray is created via the buffer interface.  It's
        # possible to distinguish shape 0 from (0, 0).
        # b = ta.array(a)
        # assert_equal(np.array(b), a)

    assert_raises(ValueError, ta.array, xrange(100000), int)
    assert_raises(ValueError, ta.array, [[[[]]]], int)
    assert_raises(TypeError, ta.array, [0, [0, 0]], int)
    assert_raises(ValueError, ta.array, [[0], [0, 0]], int)
    assert_raises(ValueError, ta.array, [[0, 0], 0], int)
    assert_raises(ValueError, ta.array, [[0, 0], [0]], int)
    assert_raises(ValueError, ta.array, [[0, 0], [[0], [0]]], int)


def test_special_constructors():
    for n in [0, 1, 2, 3, 17]:
        assert_equal(ta.zeros(n, int), np.zeros(n, int))
        assert_equal(ta.ones(n, int), np.ones(n, int))
        assert_equal(ta.identity(n, int), np.identity(n, int))


def test_dot():
    # Check acceptance of non-tinyarray arguments.
    assert_equal(ta.dot([1, 2], (3, 4)), 11)

    # The commented testcases can be added once there is support for creating
    # tinyarrays via the buffer interface.
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
    for sa, sb in shape_pairs:
        a = np.arange(np.prod(sa), dtype=int).reshape(sa)
        b = np.arange(np.prod(sb), dtype=int).reshape(sb) - 5
        assert_equal(ta.dot(ta.array(a), ta.array(b)), np.dot(a, b))

    # The commented out testcases do not work due to a bug in numpy:
    # PySequence_Check should return 0 for 0-d arrays.
    shape_pairs = [#((), 2), (2, ()),
                   (1, 2),
                   (1, (2, 2)), ((1, 1), 2),
                   ((2, 2), (3, 2)),
                   ((2, 3, 2), (4, 3)),
                   ((2, 3, 4), 3),
                   ((3, 3), (2, 4, 3)),
                   (3, (2, 4, 3)),
                   ((2, 2, 2), (2, 2, 2))]
    for sa, sb in shape_pairs:
        a = np.arange(np.prod(sa), dtype=int).reshape(sa)
        b = np.arange(np.prod(sb), dtype=int).reshape(sb) - 5
        assert_raises(ValueError, ta.dot, ta.array(a), ta.array(b))


def test_hash():
    n = 100
    assert_equal(len(set(hash(ta.array(range(i))) for i in range(n))), n)
