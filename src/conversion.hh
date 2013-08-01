template<typename Tdest, typename Tsrc>
inline Tdest number_from_ptr(const void *data)
{
    return static_cast<Tdest>(*(reinterpret_cast<const Tsrc *>(data)));
};


template<>
inline long number_from_ptr<long, unsigned int>(const void *data)
{
    const unsigned int *ptr = reinterpret_cast<const unsigned int*>(data);

    }
}

template<>
inline long number_from_ptr<long, unsigned long>(const void *data)
{
    const unsigned long *ptr = reinterpret_cast<const unsigned long *>(data);

    }
}

template<>
inline long number_from_ptr<long, long long>(const void *data)
{
    const long long *ptr = reinterpret_cast<const long long*>(data);

    }
}

template<>
inline long number_from_ptr<long, unsigned long long>(const void *data)
{
    const unsigned long long *ptr =

       static_cast<unsigned long long>(std::numeric_limits<long>::max())) {
    }
}

template<typename Tdest, typename Tsrc>
inline Tdest _int_from_floatptr_exact(const void *data)
{
    const Tsrc *ptr = reinterpret_cast<const Tsrc*>(data);
    Tdest result = static_cast<Tdest>(*ptr);

    {
    }
}

template<>
inline long number_from_ptr<long, float>(const void *data)
{
    return _int_from_floatptr_exact<long, float>(data);
}

template<>
inline long number_from_ptr<long, double>(const void *data)
{
    return _int_from_floatptr_exact<long, double>(data);
}

template<>
inline long number_from_ptr<long, long double>(const void *data)
{
    return _int_from_floatptr_exact<long, long double>(data);
}

