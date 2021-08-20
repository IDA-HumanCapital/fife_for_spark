from pyspark.sql import SparkSession
import pandas as pd
import findspark
import pyspark
from pyspark.mllib.random import RandomRDDs
import numpy as np
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import random as rn
import argparse


def create_example_data1(n_persons: int = 3, n_periods: int = 12
) -> pyspark.sql.DataFrame:
    """
    Create example data for testing FIFE

    Args:
        n_persons: the number of people to be in the dataset
        n_periods: the number of periods to be in the dataset

    Returns:
        Spark dataframe with example data
    """
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    schema = StructType([
        StructField('individual', StringType(), True),
        StructField('period', StringType(), True),
        StructField('feature_1', StringType(), True),
        StructField('feature_2', StringType(), True),
        StructField('feature_3', StringType(), True),
        StructField('feature_4', StringType(), True)])
    values = spark.createDataFrame([], schema)
    for i in np.arange(n_persons):
            period = np.random.randint(n_periods)+1 #Sparkify this
            rdd1 = RandomRDDs.uniformRDD(spark,size = 1, seed = 9999)
            x_1 = rdd1.first()

            obj1 = spark.sparkContext.parallelize(["A","B","C"])
            x_2 = obj1.takeSample(False, 1, seed = 9999)[0]

            rdd3 = RandomRDDs.uniformRDD(spark, size = 1, seed = 9999).map(lambda v: 1 + v)
            x_3 = rdd3.first()

            obj2 = spark.sparkContext.parallelize(["a","b","c", 1, 2, 3, np.nan])
            x_4 = obj2.takeSample(False, 1, seed = 9999)[0]
            while period <= n_periods:
                    print(period)
                    values = values.union(spark.createDataFrame([(int(i), period, x_1, x_2, x_3, x_4)]))
                    if x_2 == 'A':
                        unif_point1 = RandomRDDs.uniformRDD(spark, size = 1, seed = 9999).map(lambda v: (.1) * v)
                        x_1 += unif_point1.first()
                    else:
                        unif_point2 = RandomRDDs.uniformRDD(spark, size = 1, seed = 9999).map(lambda v: (.2) * v)
                        x_1 += unif_point2.first()
                    if x_1 > np.sqrt(x_3):
                        break
                    if x_4 in obj2.take(5):
                        x_4_transition_value = obj2.collect()[obj2.collect().index(x_4) + 1] #Remove all these collects
                        if RandomRDDs.uniformRDD(spark, size = 1, seed = 9999).first() >= 0.75:
                            x_4 = x_4_transition_value 
                            del x_4_transition_value
                    period += 1
    values = values.withColumn('feature_5', values.feature_2)
    return values

def create_example_data2(
    n_persons: int = 8192, n_periods: int = 20
) -> pyspark.sql.DataFrame:
    """
    Fabricate an unbalanced panel dataset suitable as FIFE input.

    Args:
        n_persons: the number of people to be in the dataset
        n_periods: the number of periods to be in the dataset

    Returns:
        Spark dataframe with example data

    """
    findspark.init()
    spark = SparkSession.builder.getOrCreate()
    seed = 9999
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

class FIFEArgParser(argparse.ArgumentParser):
    """Argument parser for the FIFE command-line interface."""

    def __init__(self):
        super().__init__()
        self.add_argument(
            "--seed",
            type=int,
            default=9999,
            help="The initializing value for all random number generators.",
        )
        self.add_argument(
            "--individual_identifier",
            type=str,
            help="The name of the feature that identifies individuals that persist over multiple time periods in the data.",
        )
        self.add_argument(
            "--time_identifier",
            type=str,
            help="The name of the feature that identifies time periods in the data.",
        )
        self.add_argument(
            "--categorical_suffixes",
            type=str,
            nargs="+",
            help="Optional list of suffixes denoting that columns ending with such a suffix should be treated as categorical.",
        )
        self.add_argument(
            "--max_null_share",
            type=float,
            help="The maximum share of observations that may have a null value for a feature to be kept for training.",
        )
        self.add_argument(
            "--max_unique_categories",
            type=int,
            help="The maximum number of unique values for a feature of a numeric type to be considered categorical.",
        )
        self.add_argument(
            "--numeric_suffixes",
            type=str,
            nargs="+",
            help="Optional list of suffixes denoting that columns ending with such a suffix should be treated as numeric.",
        )
        self.add_argument(
            "--test_intervals",
            type=int,
            help="The number of most recent periods to treat as absent from the data during training for the purpose of model evaluation.",
        )
        self.add_argument(
            "--validation_share",
            type=float,
            help="The share of observations used for evaluation instead of training for hyperoptimization or early stopping.",
        )
        self.add_argument(
            "--max_epochs",
            type=int,
            help="If HYPER_TRIALS is zero, the maximum number of passes through the training set.",
        )
        self.add_argument(
            "--min_survivors_in_train",
            type=int,
            help="The minimum number of training set observations surviving a given time horizon for the model to be trained to make predictions for that time horizon.",
        )