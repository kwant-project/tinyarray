#!/usr/bin/env python

# Copyright 2012-2016 Tinyarray authors.
#
# This file is part of Tinyarray.  It is subject to the license terms in the
# file LICENSE.rst found in the top-level directory of this distribution and
# at https://gitlab.kwant-project.org/kwant/tinyarray/blob/master/LICENSE.rst.
# A list of Tinyarray authors can be found in the README.rst file at the
# top-level directory of this distribution and at
# https://gitlab.kwant-project.org/kwant/tinyarray.

from __future__ import print_function

import subprocess
import os
import sys
import collections
from setuptools import setup, Extension, Command
from sysconfig import get_platform
from distutils.errors import DistutilsError, DistutilsModuleError
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools.command.sdist import sdist as sdist_orig
from setuptools.command.test import test as test_orig

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

try:
    from os.path import samefile
except ImportError:
    # This code path will be taken on Windows for Python < 3.2.
    # TODO: remove this once we require Python 3.2.

    def _getfinalpathname(f):
        return os.path.normcase(os.path.abspath(f))

    # This simple mockup should do in practice.
    def samefile(f1, f2):
        return _getfinalpathname(f1) == _getfinalpathname(f2)


SAVED_VERSION_FILE = 'version'


distr_root = os.path.dirname(os.path.abspath(__file__))


def configure_extensions(exts, aliases=(), build_summary=None):
    """Modify extension configuration according to the configuration file

    `exts` must be a dict of (name, kwargs) tuples that can be used like this:
    `Extension(name, **kwargs).  This function modifies the kwargs according to
    the configuration file.

    This function modifies `sys.argv`.
    """
    global config_file, config_file_present

    #### Determine the name of the configuration file.
    config_file_option = '--configfile'
    # Handle command line option
    for i, opt in enumerate(sys.argv):
        if not opt.startswith(config_file_option):
            continue
        l, _, config_file = opt.partition('=')
        if l != config_file_option or not config_file:
            print('error: Expecting {}=PATH'.format(config_file_option),
                  file=sys.stderr)
            sys.exit(1)
        sys.argv.pop(i)
        break
    else:
        config_file = 'build.conf'

    #### Read build configuration file.
    configs = configparser.ConfigParser()
    try:
        with open(config_file) as f:
            configs.readfp(f)
    except IOError:
        config_file_present = False
    else:
        config_file_present = True

    #### Handle section aliases.
    for short, long in aliases:
        if short in configs:
            if long in configs:
                print('Error: both {} and {} sections present in {}.'.format(
                    short, long, config_file))
                sys.exit(1)
            configs[long] = configs[short]
            del configs[short]

    #### Apply config from file.  Use [DEFAULT] section for missing sections.
    defaultconfig = configs.defaults()
    for name, kwargs in exts.items():
        try:
            items = configs.items(name)
        except configparser.NoSectionError:
            items = defaultconfig.items()
        else:
            configs.remove_section(name)

        for key, value in items:
            # Most, but not all, keys are lists of strings
            if key == 'language':
                pass
            elif key == 'optional':
                value = bool(int(value))
            else:
                value = value.split()

            if key == 'define_macros':
                value = [tuple(entry.split('=', 1))
                         for entry in value]
                value = [(entry[0], None) if len(entry) == 1 else entry
                         for entry in value]

            if key in kwargs:
                msg = 'Caution: user config in file {} shadows {}.{}.'
                if build_summary is not None:
                    build_summary.append(msg.format(config_file, name, key))
            kwargs[key] = value

        kwargs.setdefault('depends', []).append(config_file)

    unknown_sections = configs.sections()
    if unknown_sections:
        print('Error: Unknown sections in file {}: {}'.format(
            config_file, ', '.join(unknown_sections)))
        sys.exit(1)

    return exts


def get_version_from_git():
    try:
        p = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                             cwd=distr_root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return
    if p.wait() != 0:
        return
    if not samefile(p.communicate()[0].decode().rstrip('\n'), distr_root):
        # The top-level directory of the current Git repository is not the same
        # as the root directory of the source distribution: do not extract the
        # version from Git.
        return

    # git describe --first-parent does not take into account tags from branches
    # that were merged-in.
    for opts in [['--first-parent'], []]:
        try:
            p = subprocess.Popen(['git', 'describe', '--long'] + opts,
                                 cwd=distr_root,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return
        if p.wait() == 0:
            break
    else:
        return
    description = p.communicate()[0].decode().strip('v').rstrip('\n')

    release, dev, git = description.rsplit('-', 2)
    version = [release]
    labels = []
    if dev != "0":
        version.append(".dev{}".format(dev))
        labels.append(git)

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=distr_root)
    except OSError:
        labels.append('confused') # This should never happen.
    else:
        if p.wait() == 1:
            labels.append('dirty')

    if labels:
        version.append('+')
        version.append(".".join(labels))

    return "".join(version)


with open(os.path.join(SAVED_VERSION_FILE), 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            continue
        else:
            version = line
            break
    else:
        raise RuntimeError("Saved version file does not contain version.")
version_is_from_git = (version == "__use_git__")
if version_is_from_git:
    version = get_version_from_git()
    if not version:
        version = "unknown"


def long_description():
    text = []
    skip = True
    try:
        with open('README.rst') as f:
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


class build_ext(build_ext_orig):
    def run(self):
        with open(os.path.join('src', 'version.hh'), 'w') as f:
            f.write("// This file has been generated by setup.py.\n")
            f.write("// It is not included in source distributions.\n")
            f.write('#define VERSION "{}"\n'.format(version))
        build_ext_orig.run(self)


class sdist(sdist_orig):
    def make_release_tree(self, base_dir, files):
        sdist_orig.make_release_tree(self, base_dir, files)

        fname = os.path.join(base_dir, SAVED_VERSION_FILE)
        # This could be a hard link, so try to delete it first.  Is there any
        # way to do this atomically together with opening?
        try:
            os.remove(fname)
        except OSError:
            pass
        with open(fname, 'w') as f:
            f.write("# This file has been generated by setup.py.\n{}\n"
                    .format(version))


# The following class is based on a recipe in
# http://doc.pytest.org/en/latest/goodpractices.html#manual-integration.
class test(test_orig):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        test_orig.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        try:
            import pytest
        except:
            print('The Python package "pytest" is required to run tests.',
                  file=sys.stderr)
            sys.exit(1)
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


def main():
    exts = collections.OrderedDict([
        ('tinyarray',
         dict(language='c++',
              sources=['src/arithmetic.cc', 'src/array.cc',
                       'src/functions.cc'],
              depends=['src/arithmetic.hh', 'src/array.hh',
                       'src/conversion.hh', 'src/functions.hh']))])

    exts = configure_extensions(exts)

    classifiers = """\
    Development Status :: 5 - Production/Stable
    Intended Audience :: Science/Research
    Intended Audience :: Developers
    License :: OSI Approved :: BSD License
    Programming Language :: Python :: 2
    Programming Language :: Python :: 3
    Programming Language :: C++
    Topic :: Software Development
    Topic :: Scientific/Engineering
    Operating System :: POSIX
    Operating System :: Unix
    Operating System :: MacOS
    Operating System :: Microsoft :: Windows"""

    setup(name='tinyarray',
          version=version,
          author='Christoph Groth (CEA) and others',
          author_email='christoph.groth@cea.fr',
          description="Arrays of numbers for Python, optimized for small sizes",
          long_description=long_description(),
          url="https://gitlab.kwant-project.org/kwant/tinyarray",
          download_url="http://downloads.kwant-project.org/tinyarray/",
          license="Simplified BSD license",
          platforms=["Unix", "Linux", "Mac OS-X", "Windows"],
          classifiers=classifiers.split('\n'),
          cmdclass={'build_ext': build_ext,
                    'sdist': sdist,
                    'test': test},
          ext_modules=[Extension(name, **kwargs)
                       for name, kwargs in exts.items()])


if __name__ == '__main__':
    main()
