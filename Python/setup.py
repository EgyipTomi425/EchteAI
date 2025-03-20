from setuptools import setup, find_packages

setup(
    name="EchteAI",
    version="0.1.0",
    author="EgyipTomi425",
    author_email="tomi04252002@gmail.com",
    description="Egy mesterséges intelligencia csomag",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EgyipTomi425/EchteAI",
    packages=find_packages(where="Python/EchteAI"),
    package_dir={"": "Python"},
    install_requires=[
        "requests",
        "tqdm",
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
