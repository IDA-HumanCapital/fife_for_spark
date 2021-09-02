from pyspark.sql import SparkSession
import pandas as pd
import findspark
import pyspark
import numpy as np
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import random as rn
import argparse

def create_example_data1(n_persons: int = 1000, n_periods: int = 20
) -> pyspark.sql.DataFrame:
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    schema = StructType([
        StructField('individual', IntegerType(), True),
        StructField('period', IntegerType(), True),
        StructField('feature_1', FloatType(), True),
        StructField('feature_2', StringType(), True),
        StructField('feature_3', IntegerType(), True),
        StructField('feature_4', StringType(), True)])
    values = spark.createDataFrame([], schema)
    for i in np.arange(n_persons):
            period = np.random.randint(n_periods) + 1
            x_1 = np.random.uniform()
            x_2 = rn.choice(["A", "B", "C"])
            x_3 = np.random.uniform() + 1.0
            x_4_categories = [1, 2, 3, 'a', 'b', 'c', np.nan]
            x_4 = rn.choice(x_4_categories)
            while period <= n_periods:
                values = values.union(spark.createDataFrame([(int(i), period, x_1, x_2, x_3, x_4)]))
                if x_2 == "A":
                    x_1 += np.random.uniform(0, 0.1)
                else:
                    x_1 += np.random.uniform(0, 0.2)
                if x_1 > np.sqrt(x_3):
                    break
                if x_4 in x_4_categories[:-2]:
                    x_4_transition_value = x_4_categories[x_4_categories.index(x_4) + 1]
                    if np.random.uniform() >= 0.75:
                        x_4 = x_4_transition_value
                        del x_4_transition_value
                period += 1
    values = values.withColumn('feature_5', values.feature_2)
    return values

def create_example_data2(
    n_persons: int = 8192, n_periods: int = 20, seed_value:int = 9999
) -> pyspark.sql.DataFrame:
    """
    Fabricate an unbalanced panel dataset suitable as FIFE input.

    Args:
        n_persons: the number of people to be in the dataset
        n_periods: the number of periods to be in the dataset
        seed_value: seed for random value generation

    Returns:
        Spark dataframe with example data

    """
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    seed = seed_value
    np.random.seed(seed)
    values = []
    for i in np.arange(n_persons):
        period = np.random.randint(n_periods) + 1
        x_1 = np.random.uniform()
        x_2 = rn.choice(["A", "B", "C"])
        x_3 = np.random.uniform() + 1.0
        #Pyspark RDD does not support a column with mutliple dtypes (both string and int)
        x_4_categories = [1, 2, 3, 'a', 'b', 'c', np.nan]
        x_4 = rn.choice(x_4_categories)
        while period <= n_periods:
            values.append([i, period, x_1, x_2, x_3, x_4])
            if x_2 == "A":
                x_1 += np.random.uniform(0, 0.1)
            else:
                x_1 += np.random.uniform(0, 0.2)
            if x_1 > np.sqrt(x_3):
                break
            if x_4 in x_4_categories[:-2]:
                x_4_transition_value = x_4_categories[x_4_categories.index(x_4) + 1]
                if np.random.uniform() >= 0.75:
                    x_4 = x_4_transition_value
                    del x_4_transition_value
            period += 1
    values = pd.DataFrame(
        values,
        columns=[
            "individual",
            "period",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ],
    )
    values["feature_5"] = values["feature_2"]
    schema = StructType([
        StructField('individual', IntegerType(), True),
        StructField('period', IntegerType(), True),
        StructField('feature_1', FloatType(), True),
        StructField('feature_2', StringType(), True),
        StructField('feature_3', FloatType(), True),
        StructField('feature_4', StringType(), True),
        StructField('feature_5', StringType(), True)])
    return spark.createDataFrame(values, schema = schema)

def import_data_file(path: str = "Input Data") -> pyspark.sql.DataFrame:
    """ Read data into a distributed spark dataframe.. 
    """
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    
    if path is not None:
        if path.endswith(".avro"):
            data = spark.read.format('avro').load(path)
        elif path.endswith(".csv"):
            data = spark.read.format('csv').load(path)
        elif path.endswith('.txt'):
            data = spark.read.format('text').load(path)
        elif path.endswith(".json"):
            data = spark.read.format('json').load(path)
        elif path.endswith(".parquet"):
            data = spark.read.load(path)
        else:
            raise Exception(
                "Data file extension is invalid.")
    return data

class FIFEArgParser(argparse.ArgumentParser):
    """Argument parser for the FIFE command-line interface."""

    def __init__(self):
        super().__init__()
        self.add_argument(
            "--SEED",
            type=int,
            default=9999,
            help="The initializing value for all random number generators.",
        )
        self.add_argument(
            "--INDIVIDUAL_IDENTIFIER",
            type=str,
            help="The name of the feature that identifies individuals that persist over multiple time periods in the data.",
        )
        self.add_argument(
            "--TIME_IDENTIFIER",
            type=str,
            help="The name of the feature that identifies time periods in the data.",
        )
        self.add_argument(
            "--CATEGORICAL_SUFFIXES",
            type=str,
            nargs="+",
            help="Optional list of suffixes denoting that columns ending with such a suffix should be treated as categorical.",
        )
        self.add_argument(
            "--MAX_NULL_SHARE",
            type=float,
            help="The maximum share of observations that may have a null value for a feature to be kept for training.",
        )
        self.add_argument(
            "--MAX_UNIQUE_CATEGORIES",
            type=int,
            help="The maximum number of unique values for a feature of a numeric type to be considered categorical.",
        )
        self.add_argument(
            "--NUMERIC_SUFFIXES",
            type=str,
            nargs="+",
            help="Optional list of suffixes denoting that columns ending with such a suffix should be treated as numeric.",
        )
        self.add_argument(
            "--TEST_INTERVALS",
            type=int,
            help="The number of most recent periods to treat as absent from the data during training for the purpose of model evaluation.",
        )
        self.add_argument(
            "--VALIDATION_SHARE",
            type=float,
            help="The share of observations used for evaluation instead of training for hyperoptimization or early stopping.",
        )
        self.add_argument(
            "--MAX_EPOCHS",
            type=int,
            help="If HYPER_TRIALS is zero, the maximum number of passes through the training set.",
        )
        self.add_argument(
            "--MIN_SURVIVOR_IN_TRAIN",
            type=int,
            help="The minimum number of training set observations surviving a given time horizon for the model to be trained to make predictions for that time horizon.",
        )
        self.add_argument(
            "--CACHE",
            type=bool,
            help="A boolean value indicating whether or not to cache designated objects. Caching improves algorithm runtime, but is not applicable on larger datasets due to memory errors.",
        )