import os
import sys

sys.path.insert(0, os.path.abspath("../"))
project = "FIFE For Spark"
copyright = "2021 - 2022, Institute for Defense Analyses"
author = "Institute for Defense Analyses"
release = "0.0.1"
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "m2r"]
html_theme = "sphinx_rtd_theme"
source_suffix = [".rst", ".md"]
autodoc_mock_imports = [
    "pyspark",
    "findspark",
    "mmlspark",
    "lifelines",
    "pandas",
    "databricks",
    "numpy",
    "fifeforspark.survival_modeler",
    "typing",
]
