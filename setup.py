"""Setup script for learning_to_see package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="learning_to_see",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Vision-language learning on synthetic scenes with compositional generalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/learning_to_see",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    package_data={
        "dsl": ["vocab.json"],
    },
    include_package_data=True,
)
