"""Standalone build script for the ezgatr.opt._opt_ops C++ extension.

Run as:

    python setup.py build_ext --inplace

This is intentionally not wired into pyproject.toml — the project uses conda
for environment/dependency management and this script's only job is to compile
the extension into src/ezgatr/opt/ where the editable install picks it up.
"""
import os

from setuptools import setup
from setuptools.dist import Distribution
from torch.utils.cpp_extension import BuildExtension, CppExtension


class StandaloneExtensionDistribution(Distribution):
    def parse_config_files(self, filenames=None, ignore_option_errors=False):
        return None


# --- optional build knobs (used by the compiler / autovectorization benchmarks) ---
# EZGATR_NO_VECTORIZE=1  -> disable GCC/Clang auto-vectorization (scalar codegen for
#                          the v0-v2 C++ kernels; v3 is explicit intrinsics, unaffected).
# EZGATR_EXTRA_CXXFLAGS  -> space-separated flags appended verbatim (e.g. "-O2").
# The compiler is chosen via the standard CXX/CC env vars, which
# torch.utils.cpp_extension honors (e.g. `CXX=clang++ CC=clang python setup.py ...`).
_extra = ["-O3", "-std=c++17", "-march=native", "-fvisibility=hidden", "-Wall", "-Wextra"]
if os.environ.get("EZGATR_NO_VECTORIZE") == "1":
    _extra += ["-fno-tree-vectorize", "-fno-tree-slp-vectorize"]
if os.environ.get("EZGATR_EXTRA_CXXFLAGS"):
    _extra += os.environ["EZGATR_EXTRA_CXXFLAGS"].split()


ext = CppExtension(
    name="ezgatr.opt._opt_ops",
    sources=[
        "src/ezgatr/_csrc/attention_ops.cpp",
        "src/ezgatr/_csrc/pga_ops.cpp",
        "src/ezgatr/_csrc/rms_ops.cpp",
        "src/ezgatr/_csrc/bindings.cpp",
    ],
    include_dirs=["src/ezgatr/_csrc"],
    extra_compile_args=_extra,
    extra_link_args=[],
    define_macros=[
        ("EZGATR_OPT_VERSION", '"0.1.0"'),
    ],
)

setup(
    name="ezgatr-opt-ext",
    package_dir={"": "src"},
    distclass=StandaloneExtensionDistribution,
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
