# -*- coding: utf-8 -*-
import setuptools

from setuptools.extension import Extension
from os import linesep
from pathlib import Path

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from numpy import get_include as numpy_get_include


# this testing macro should always be set to 0/False
# except during active developement. TESTING sets
# LOGGING to verbose automatically
TESTING = 1  # determines how .pyx files are compiled
LOGGING = 0  # higher values indicate verbosity


class MyBuildExt(build_ext):
    def run(self):
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent

        target_dir = build_dir if not self.inplace else root_dir
        init_path = target_dir / 'cnn' / '__init__.py'
        lines = (
            "from .dense import NN",
            "from .data import preprocess_data",
        )

        init_path.touch()
        with open(init_path, 'w') as f:
            f.write(linesep.join(lines))


def setup():
    def _extension_named(name):
        return Extension(
            f"cnn.{name}",
            [f"src/cnn/{name}.pyx"],
            define_macros=[("NPY_NO_DEPRECATED_API", None)],
            include_dirs=[numpy_get_include(),],
            #language='c++',
        )

    setuptools.setup(
        name='cnn',
        version='0.0.1',
        author_email="spencer.hann@gmail.com",
        url="https://github.com/spencer-hann/CNN.git",
        package_dir={"":"src"},
        ext_modules=cythonize(
            [_extension_named('*')],
            compile_time_env={'TESTING':TESTING, 'LOGGING':LOGGING},
            #force=True,
        ),
        cmdclass=dict(build_ext=MyBuildExt),
        install_requires=['numpy'],
    )


if __name__ == "__main__":
    setup()

