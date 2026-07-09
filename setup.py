from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sparse-kappa",                    
    version="0.0.4",
    author="Erin Carson and Xinye Chen",
    author_email="carson@karlin.mff.cuni.cz; xinyechenai@gmail.com",
    description="Algorithms for sparse matrix condition number estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inEXASCALE/sparse-kappa",
    project_urls={
        "Bug Tracker": "https://github.com/inEXASCALE/sparse-kappa/issues",
    },
    packages=find_packages(include=["sparse_kappa", "sparse_kappa.*"]),
    package_dir={"": "."},
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "myst-parser>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
)
