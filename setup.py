from setuptools import setup, find_packages

setup(
    name="synergy-models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.2.0",
        "seaborn>=0.11.0",
        "scipy>=1.4.0",
        "statsmodels>=0.12.0",
        "tqdm>=4.46.0",
    ],
    author="Leonardo Lonati",
    author_email="leonardo.lonati@unipv.it",
    description="A package for analyzing drug synergy using the ZIP model",
    keywords="synergy, drug-combinations, statistical-modeling, ZIP-model",
    url="https://github.com/LLonati/synergy-models",
)