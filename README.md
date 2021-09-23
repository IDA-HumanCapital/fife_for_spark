The Finite-Interval Forecasting Engine for Spark (FIFEforSpark) is an adaptation of the Finite-Interval Forecasting Engine for the Apache Spark environment. Currently, it provides machine learning models, specifically a gradient boosted tree model, for discrete-time survival analysis.

If you are already familiar with FIFE, you'll recognize the following explanation of how FIFEforSpark approaches survival analysis. Many of the sections were borrowed heavily from FIFE as this is merely an adaptation of the package to the Spark environment with the exact same methodology. If you would like more information on FIFE, you can read the documentation [here](https://fife.readthedocs.io/en/latest). If you want more documentation on FIFEforSpark, you can go [here](https://fife-for-spark.readthedocs.io/en/latest/)

Suppose you have a dataset that looks like this:

| ID | period | feature_1 | feature_2 | feature_3 | ... |
|----|--------|-----------|-----------|-----------|-----|
| 0  | 2016   | 7.2       | A         | 2AX       | ... |
| 0  | 2017   | 6.4       | A         | 2AX       | ... |
| 0  | 2018   | 6.6       | A         | 1FX       | ... |
| 0  | 2019   | 7.1       | A         | 1FX       | ... |
| 1  | 2016   | 5.3       | B         | 1RM       | ... |
| 1  | 2017   | 5.4       | B         | 1RM       | ... |
| 2  | 2017   | 6.7       | A         | 1FX       | ... |
| 2  | 2018   | 6.9       | A         | 1RM       | ... |
| 2  | 2019   | 6.9       | A         | 1FX       | ... |
| 3  | 2017   | 4.3       | B         | 2AX       | ... |
| 3  | 2018   | 4.1       | B         | 2AX       | ... |
| 4  | 2019   | 7.4       | B         | 1RM       | ... |
| ...| ...    | ...       | ...       |...        | ... |

The entities with IDs 0, 2, and 4 are observed in the dataset in 2019.

While FIFE offers a significantly larger suite of models designed to answer a variety of questions, FIFEforSpark is mainly focused on one question: what are each individual's probabilities of being observed in any future year? Fortunately, FIFEforSpark can estimate answers to these questions for any unbalanced panel dataset.

Exactly like FIFE, FIFEforSpark unifies survival analysis and multivariate time series analysis. Tools for the former neglect future states of survival; tools for the latter neglect the possibility of discontinuation. Traditional forecasting approaches for each, such as proportional hazards and vector autoregression (VAR), respectively, impose restrictive functional forms that limit forecasting performance. FIFEforSpark supports one of the best approaches for maximizing forecasting performance: gradient-boosted trees (using MMLSpark's LightGBM).

FIFEforSpark is simple to use and the syntax is almost identical to that of FIFE; however, given that this is meant to be run in the Spark environment in a Python notebook, there are some notable differences. First, the package 'mmlspark' must already be installed and attached to the cluster. Unfortunately, the PyPI version of MMLSpark is not compatible with FIFEforSpark. As such, FIFE is best utilized in a Databricks notebook. For a tutorial on how to download mmlspark on databricks, click [here](https://fife-for-spark.readthedocs.io/en/latest/spark_help.html).

FIFEforSpark is a supported package on PyPI (Python Package Index), thus downloading FIFEforSpark is as simple as entering the package name in the 'Create Library' tab on Databricks (with Library Source set to PyPI) or by running the following statement in the command prompt:

```console
pip install fifeforspark 
```

Once installed, generating forecasts is easy. If you are working in a Databricks python notebook, you may run something like the following code, where 'your_table' is the name of your table.

```python
from fifeforspark.processors import PanelDataProcessor
from fifeforspark.lgb_modelers import LGBSurvivalModeler

data_processor = PanelDataProcessor(data = spark.sql("select * from your_table"))
data_processor.build_processed_data()

modeler = LGBSurvivalModeler(data=data_processor.data)
modeler.build_model()

forecasts = modeler.forecast()
```

If you are working in a Python IDE and have both pyspark and MMLSpark successfully installed, you can run the following:

```python
import findspark
findspark.init()
import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession

from fifeforspark.processors import PanelDataProcessor
from fifeforspark.lgb_modelers import LGBSurvivalModeler

spark = SparkSession.builder.getOrCreate()
data_processor = PanelDataProcessor(data=spark.read.csv(path_to_your_data))
data_processor.build_processed_data()

modeler = LGBSurvivalModeler(data=data_processor.data)
modeler.build_model()

forecasts = modeler.forecast()
```


Here's a notebook with real data, where we forecast when world leaders will lose power: [REIGN Example Notebook](https://nbviewer.jupyter.org/github/IDA-HumanCapital/fife_for_spark/blob/main/examples/example_reign_notebook.ipynb)

If you would like more information on FIFEforSpark, you can read the documentation here: [FIFEforSpark Documentation](https://fife-for-spark.readthedocs.io/en/latest/)
