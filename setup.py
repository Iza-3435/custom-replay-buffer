"""
Setup script for High-Performance Replay Buffer Library
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import os
import sys

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "High-Performance Replay Buffer Library for Reinforcement Learning"

# Get version
def get_version():
    version_path = os.path.join(os.path.dirname(__file__), 'python', 'replay_buffer', '__init__.py')
    if os.path.exists(version_path):
        with open(version_path, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

# C++ extension modules (optional - for when we have full C++ bindings)
ext_modules = []

# Check if we should build C++ extensions
build_cpp = os.environ.get('BUILD_CPP', 'false').lower() == 'true'

if build_cpp:
    try:
        # Define C++ extension
        cpp_extension = Pybind11Extension(
            "replay_buffer._cpp_core",
            sources=[
                "src/replay_buffer/core/simple_buffer.cpp",
                "src/replay_buffer/core/uniform_buffer.cpp",
                # Add more source files as needed
            ],
            include_dirs=[
                "include",
                # Add other include directories
            ],
            language='c++',
            cxx_std=17,
        )
        ext_modules.append(cpp_extension)
    except ImportError:
        print("Warning: pybind11 not available, skipping C++ extensions")

setup(
    name="high-performance-replay-buffer",
    version=get_version(),
    author="HFT Systems",
    author_email="contact@hftsystems.ai",
    description="Ultra-fast C++ replay buffer implementation for reinforcement learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/high-performance-replay-buffer",
    
    # Package configuration
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    
    # Include package data
    package_data={
        "replay_buffer": ["*.so", "*.dll", "*.dylib"],  # Include compiled libraries
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=[
        "numpy>=1.19.0",
        "dataclasses; python_version<'3.7'",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "examples": [
            "matplotlib",
            "jupyter",
        ],
        "cpp": [
            "pybind11>=2.6.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.6",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
    ],
    
    # Entry points
    entry_points={
        "console_scripts": [
            "replay-buffer-benchmark=replay_buffer.utils:benchmark_performance",
        ],
    },
    
    # C++ extensions
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext} if build_cpp else {},
    zip_safe=False,
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/high-performance-replay-buffer/issues",
        "Source": "https://github.com/yourusername/high-performance-replay-buffer",
        "Documentation": "https://github.com/yourusername/high-performance-replay-buffer/docs",
    },
)