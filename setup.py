#!/usr/bin/env python

# Copyright 2012-2013 Tinyarray authors.
#
# This file is part of Tinyarray.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://git.kwant-project.org/tinyarray/about/LICENSE.  A list of Tinyarray
# authors can be found in the README file at the top-level directory of this
# distribution and at http://git.kwant-project.org/tinyarray/about/.

import subprocess
import os
import sys
from distutils.core import setup, Extension, Command
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsModuleError

README_FILE = 'README'
STATIC_VERSION_FILE = 'src/version.hh'
TEST_MODULE = 'test_tinyarray.py'

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: C++
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Operating System :: Microsoft :: Windows"""

tinyarray_dir = os.path.dirname(os.path.abspath(__file__))


def get_version_from_git():
    try:
        p = subprocess.Popen(['git', 'describe'], cwd=tinyarray_dir,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return

    if p.wait() != 0:
        return
    version = p.communicate()[0].strip()

    if version[0] == 'v':
        version = version[1:]

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=tinyarray_dir)
    except OSError:
        version += '-confused'  # This should never happen.
    else:
        if p.wait() == 1:
            version += '-dirty'
    return version


def get_static_version():
    """Return the version as recorded inside a file."""
    try:
        with open(STATIC_VERSION_FILE) as f:
            contents = f.read()
            assert contents[:17] == '#define VERSION "'
            assert contents[-2:] == '"\n'
            return contents[17:-2]
    except:
        return None


def version():
    """Determine the version of Tinyarray.  Return it and save it in a file."""
    git_version = get_version_from_git()
    static_version = get_static_version()
    if git_version is not None:
        version = git_version
        if static_version != git_version:
            with open(STATIC_VERSION_FILE, 'w') as f:
                f.write('#define VERSION "{}"\n'.format(version))
    elif static_version is not None:
        version = static_version
    else:
        version = 'unknown'
    return version


def long_description():
    text = []
    skip = True
    try:
        with open(README_FILE) as f:
            for line in f:
                if line == "\n":
                    if skip:
                        skip = False
                        continue
                    elif text[-1] == '\n':
                        text.pop()
                        break
                if not skip:
                    text.append(line)
    except:
        return ''
    text[-1] = text[-1].rstrip()
    return ''.join(text)


class test(Command):
    description = "run the unit tests"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            from nose.core import run
        except ImportError:
            raise DistutilsModuleError('nose <http://nose.readthedocs.org/> '
                                       'is needed to run the tests')
        self.run_command('build')
        major, minor = sys.version_info[:2]
        lib_dir = "build/lib.{0}-{1}.{2}".format(get_platform(), major, minor)
        print '**************** Tests ****************'
        if not run(argv=[__file__, '-v', '-w', lib_dir,
                         '-w', '../../' + TEST_MODULE]):
            raise DistutilsError('at least one of the tests failed')


module = Extension('tinyarray',
                   language='c++',
                   sources=['src/arithmetic.cc', 'src/array.cc',
                            'src/functions.cc'],
                   depends=['src/arithmetic.hh', 'src/array.hh',
                            'src/conversion.hh', 'src/functions.hh'])


def main():
    setup(name='tinyarray',
          version=version(),
          author='Christoph Groth (CEA) and others',
          author_email='christoph.groth@cea.fr',
          description="Arrays of numbers for Python, optimized for small sizes",
          long_description=long_description(),
          url="http://kwant-project.org/tinyarray/",
          download_url="http://downloads.kwant-project.org/tinyarray/",
          license="Simplified BSD license",
          platforms=["Unix", "Linux", "Mac OS-X", "Windows"],
          classifiers=CLASSIFIERS.split('\n'),
          cmdclass={'test': test},
          ext_modules=[module])


if __name__ == '__main__':
    main()
