"""Standalone build script for the ezgatr.opt._opt_ops C++ extension.

Run as:

    python setup.py build_ext --inplace

This is intentionally not wired into pyproject.toml — the project uses conda
for environment/dependency management and this script's only job is to compile
the extension into src/ezgatr/opt/ where the editable install picks it up.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext = CppExtension(
    name="ezgatr.opt._opt_ops",
    sources=[
        "src/ezgatr/_csrc/attention_ops.cpp",
        "src/ezgatr/_csrc/pga_ops.cpp",
        "src/ezgatr/_csrc/rms_ops.cpp",
        "src/ezgatr/_csrc/bindings.cpp",
    ],
    include_dirs=["src/ezgatr/_csrc"],
    extra_compile_args=[
        "-O3",
        "-std=c++17",
        "-march=native",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
    ],
    extra_link_args=[],
    define_macros=[
        ("EZGATR_OPT_VERSION", '"0.1.0"'),
    ],
)

setup(
    name="ezgatr-opt-ext",
    package_dir={"": "src"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
