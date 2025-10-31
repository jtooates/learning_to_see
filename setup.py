from setuptools import setup, find_packages

setup(
    name="shape-scene-generator",
    version="0.1.0",
    description="Synthetic 2D shape scene generator for vision-language ML",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ]
    },
    python_requires=">=3.8",
)
