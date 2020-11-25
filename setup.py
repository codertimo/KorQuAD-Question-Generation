from setuptools import find_packages, setup

setup(
    name="korquad-qg",
    version="0.0.1",
    description="Question Generation Model with KorQuAD",
    install_requires=[],
    url="https://github.com/codertimo/KorQuAD-Question-Generation",
    author="codertimo",
    author_email="codertimo@gmail.com",
    packages=find_packages(exclude=["tests", "scripts"]),
)
