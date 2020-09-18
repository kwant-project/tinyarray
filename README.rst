Tinyarray
=========

Tinyarrays are similar to NumPy arrays, but optimized for small sizes.
Common operations on very small arrays are to 3-7 times faster than with
NumPy (with NumPy 1.6 it used to be up to 35 times), and 3 times less
memory is used to store them.  Tinyarrays are useful if you need many
small arrays of numbers, and cannot combine them into a few large ones.
(The resulting code is still much slower than C, but it may now be fast
enough.)

Unlike Python's built-in tuples, Tinyarrays support mathematical
operations like element-wise addition and matrix multiplication.  Unlike
Numpy arrays, Tinyarrays can be used as dictionary keys because they are
hashable and immutable.  What is more, tinyarrays are equivalent to
tuples with regard to hashing and comparisons: a dictionary or set with
tinyarray keys may by transparently indexed by tuples.

The module's interface is a subset of that of NumPy and thus should be
familiar to many.  Whenever an operation is missing from Tinyarray,
NumPy functions can be used directly with Tinyarrays.


Tinyarray is licensed under the "simplified BSD License".  See
`<LICENSE.rst>`_.

Website: https://gitlab.kwant-project.org/kwant/tinyarray

Please report bugs here:
https://gitlab.kwant-project.org/kwant/tinyarray/issues


Source code
-----------

Source tarballs are available at http://downloads.kwant-project.org/tinyarray/

Clone the Git repository with ::

    git clone https://gitlab.kwant-project.org/kwant/tinyarray.git


Installation
------------

Tinyarray can be built from source with the usual ::

    pip install tinyarray

One can also download the source tarball (or clone it from git) and use ::

    python setup.py install

Prepared packages exist for

* Windows

  Use `Christoph Gohlke's installer
  <http://www.lfd.uci.edu/~gohlke/pythonlibs/#tinyarray>`_.

* Ubuntu and derivatives ::

      sudo apt-add-repository ppa:kwant-project/ppa
      sudo apt-get update
      sudo apt-get install python-tinyarray

* Debian and derivatives

  1. Add the following lines to ``/etc/apt/sources.list``::

         deb http://downloads.kwant-project.org/debian/ stable main
         deb-src http://downloads.kwant-project.org/debian/ stable main

  2. (Optional) Add the OpenPGP key used to sign the repositories by executing::

         sudo apt-key adv --keyserver pgp.mit.edu --recv-key C3F147F5980F3535

  3. Update the package data, and install::

         sudo apt-get update
         sudo apt-get install python-tinyarray

* Mac OS X

  Follow the `instructions for installing "Kwant"
  <http://kwant-project.org/install#mac-os-x>`_ but install
  ``py27-tinyarray`` instead of ``py27-kwant`` etc.


Build configuration
-------------------

If necessary, the compilation and linking of tinyarray can be configured with
a build configuration file.  By default, this file is ``build.conf`` in the
root directory of the tinyarray distribution.  A different path may be
provided using the ``--configfile=PATH`` option.

The configuration file consists of sections, one for each extension module
(currently there is only one: ``tinyarray``), led by a ``[section name]``
header and followed by ``key = value`` lines.

Possible keys are the keyword arguments for ``distutils.core.Extension`` (For
a complete list, see its `documentation
<https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension>`_).
The corresponding values are whitespace-separated lists of strings.

Example ``build.conf`` for compiling Tinyarray with C assertions::

    [tinyarray]
    undef_macros = NDEBUG


Usage example
-------------

The following example shows that in simple cases Tinyarray works just as
NumPy. ::

    from math import sin, cos, sqrt
    import tinyarray as ta

    # Make a vector.
    v = ta.array([1.0, 2.0, 3.0])

    # Make a rotation matrix.
    alpha = 0.77
    c, s = cos(alpha), sin(alpha)
    rot_z = ta.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])

    # Rotate the vector, normalize, and print it.
    v = ta.dot(rot_z, v)
    v /= sqrt(ta.dot(v, v))
    print v


Documentation
-------------

The module's interface is a basic subset of NumPy and hence should be familiar
to many Python programmers.  All functions are simplified versions of their
NumPy counterparts.  The module's docstring serves as main documentation.  To
see it, run in Python::

    import tinyarray as ta
    help(ta)

Or in the system shell::

    pydoc tinyarray


Contributing
------------

Contributions to tinyarray are most welcome.  Patches may be sent by email, or
a merge request may be opened on the Project's website.

Please add tests for any new functionality and make sure that all existing
tests still run.  To run the tests, execute::

    python setup.py test

It is a good idea to enable C assertions as shown above under
`Build configuration`_.


Authors
-------

The principal developer of Tinyarray is Christoph Groth (CEA
Grenoble).  His contributions are part of his work at `CEA <http://cea.fr/>`_,
the French Commissariat à l'énergie atomique et aux énergies alternatives.

The author can be reached at christoph.groth@cea.fr.

Other people that have contributed to Tinyarray include

* Michael Wimmer (Leiden University, TU Delft)
* Joseph Weston (CEA Grenoble, TU Delft)
* Jörg Behrmann (FU Berlin)
