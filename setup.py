"""Setup for FIFEforSpark: Finite-Interval Forecasting Engine for Spark."""

from setuptools import setup

with open("README.md", "r") as f:
    README = f.read()

setup(
    name="fifeforspark",
    version="0.0.1",
    description=(
        "Finite-Interval Forecasting Engine for Spark: Machine learning models "
        "for discrete-time survival analysis and multivariate time series "
        "forecasting for Apache Spark"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/IDA-HumanCapital/fifeforspark",
    #project_urls={
    #    "Bug Tracker": "https://github.com/IDA-HumanCapital/fife/issues",
    #    "Source Code": "https://github.com/IDA-HumanCapital/fife",
    #    "Documentation": "https://fife.readthedocs.io/en/latest",
    #},
    author="Institute for Defense Analyses",
    author_email="humancapital@ida.org",
    license="AGPLv3+",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: "
        "GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=["fife"],
    install_requires=[
        "ipython",
        "lifelines",
        "pyspark",
        "findspark",
        "numpy",
        "pandas",
        "databricks"
    ],
    #extras_require={"shap": ["shap"]},
    entry_points={
        "console_scripts": [
            "fifeforspark=fifeforspark.__main__:main",
        ]
    },
)
