from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import numpy as np
import sys
import os

# Get the directory containing this setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(setup_dir, "..", ".."))

# Determine Eigen include directory based on build directory
eigen_include = os.path.join(project_root, "build", "_deps", "eigen-src")
if not os.path.exists(eigen_include):
    # Fallback to system Eigen or download location
    eigen_include = "/usr/include/eigen3"

ext_modules = [
    Pybind11Extension(
        "pyhnsw",
        ["src/pyhnsw.cpp"],
        include_dirs=[
            os.path.join(project_root, "include"),
            eigen_include,
            np.get_include(),
        ],
        cxx_std=17,
        define_macros=[("EIGEN_NO_DEBUG", None)],
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
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python bindings for HNSW (Hierarchical Navigable Small World) vector search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
)