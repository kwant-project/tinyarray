import sys
            assert_equal(memoryview(b).tobytes(),
                         memoryview(a).tobytes())
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
            assert_equal(memoryview(b).tobytes(),
                         memoryview(a).tobytes())
            assert_equal(ta.transpose(b), np.transpose(a))

            if not isinstance(a_shape, tuple) or len(a_shape) <= 2:
                b = ta.array(np.matrix(a))

                assert_equal(b.ndim, 2)
                assert_equal(b, np.matrix(a))
        a = ta.matrix(b)
        assert_equal(a, b)

            npsrc = np.array(tsrc)
            for s in [src, tsrc, npsrc]:
    long_overflow = [1e300, np.array([1e300])]
    if 18446744073709551615 > sys.maxint:
        long_overflow.extend([np.array([18446744073709551615], np.uint64),
                              18446744073709551615])

    for s in long_overflow:
        assert_raises(OverflowError, ta.array, s, long)

            assert_raises(ValueError, ta.dot, ta.array(a.tolist()),
                          ta.array(b.tolist()))
            assert_equal(ta.array(a) + ta.array(b), a + b)
            assert_equal(ta.array(a) + ta.array(b), a + b)
                    assert_equal(op(ta.array(a), ta.array(b)), op(a, b))
                    assert_equal(ta_func(a, b), np_func(a, b))
                assert_equal(op(ta.array(a)), op(a))


def test_pickle():
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
