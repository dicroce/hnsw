from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import numpy as np
import sys
import os
import subprocess
import tempfile

# Get the directory containing this setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(setup_dir, "..", ".."))

# Determine Eigen include directory
eigen_include_paths = [
    os.path.join(project_root, "build", "_deps", "eigen-src"),  # CMake FetchContent location
    "/usr/include/eigen3",  # Ubuntu/Debian
    "/usr/local/include/eigen3",  # Homebrew on macOS
    "/opt/homebrew/include/eigen3",  # Homebrew on Apple Silicon
]

eigen_include = None
for path in eigen_include_paths:
    if os.path.exists(path):
        eigen_include = path
        break

# If no system Eigen found, we'll fetch it during build
if eigen_include is None:
    # Create a temporary directory for Eigen
    temp_dir = tempfile.mkdtemp()
    eigen_include = os.path.join(temp_dir, "eigen")
    
    # Download and extract Eigen with better error handling
    try:
        print("Downloading Eigen...")
        result = subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "3.4.0",
            "https://gitlab.com/libeigen/eigen.git", eigen_include
        ], check=True, capture_output=True, text=True, timeout=300)
        print(f"Successfully downloaded Eigen to {eigen_include}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Failed to download Eigen: {e}")
        # Try alternative: download as tarball
        try:
            import urllib.request
            import tarfile
            print("Trying alternative download method...")
            tarball_url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
            tarball_path = os.path.join(temp_dir, "eigen.tar.gz")
            urllib.request.urlretrieve(tarball_url, tarball_path)
            with tarfile.open(tarball_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
            eigen_include = os.path.join(temp_dir, "eigen-3.4.0")
            print(f"Successfully downloaded Eigen tarball to {eigen_include}")
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            # Last resort: use empty path and hope system has it
            eigen_include = ""
            print("Warning: Could not locate or download Eigen. Hoping system has it in standard paths.")

ext_modules = [
    Pybind11Extension(
        "pyhnsw",
        ["src/pyhnsw.cpp"],
        include_dirs=[
            path for path in [
                os.path.join(project_root, "include"),
                eigen_include,
                np.get_include(),
            ] if path  # Filter out empty paths
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