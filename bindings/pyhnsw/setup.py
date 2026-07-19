from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np
import os

# Get the directory containing this setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(setup_dir, "..", ".."))

# The extension is dependency-free (header-only hnsw + numpy for the array glue).
# The SIMD distance kernels select an AVX2/SSE2/scalar path at RUNTIME, so we
# deliberately do NOT pass -march=native / /arch:AVX2: the wheel must run on any
# x86-64 CPU. Pybind11Extension's default flags are already a safe baseline.
ext_modules = [
    Pybind11Extension(
        "pyhnsw",
        ["src/pyhnsw.cpp"],
        include_dirs=[
            os.path.join(project_root, "include"),
            np.get_include(),
        ],
        cxx_std=17,
    ),
]

# Read long description from README if it exists
long_description = ""
readme_path = os.path.join(setup_dir, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="pyhnsw",
    author="Tony Di Croce",
    author_email="dicroce@gmail.com",
    description="Python bindings for HNSW (Hierarchical Navigable Small World) vector search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19.0",
    ],
    setup_requires=[
        "pybind11>=2.6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
)