import findspark
import pyspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import isnan, lit, col, lag
from pyspark.sql.types import DateType, TimestampType, StringType, IntegerType, LongType, ShortType, ByteType, FloatType, DoubleType, DecimalType
import pyspark.sql.functions as F

findspark.init()
sc = SparkSession.builder.getOrCreate()

def check_column_consistency(spark_df: pyspark.sql.DataFrame, colname: str) -> None:
    """Assert column exists, has no missing values, and is not constant."""
    assert colname in spark_df.columns, f"{colname} not in data"
    assert spark_df.select(isnan(colname).cast('int').alias(colname)).agg({colname:'sum'}).first()[0] == 0, f"{colname} has missing values"
    assert spark_df.select(colname).distinct().count() >= 2, f"{colname} does not have multiple unique values"
    
def check_panel_consistency(spark_df: pyspark.sql.DataFrame, individual_identifier: str, time_identifier: str) -> None:
    """Ensure observations have unique individual-period combinations."""
    check_column_consistency(spark_df, individual_identifier)
    check_column_consistency(spark_df, time_identifier)
    subset = spark_df.select([individual_identifier, time_identifier])
    assert subset.count() == subset.dropDuplicates().count(), "One or more individuals have multiple observations for a single time value"
    
def is_degenerate(spark_df: pyspark.sql.DataFrame , col: str, max_null_share: float = .999) -> bool:
    """Determine if a feature is constant or has too many missing values."""
    if spark_df.select(isnan(col).cast('integer').alias(col)).agg({col:'mean'}).first()[0] >= max_null_share:
        return True
    if spark_df.select(col).distinct().count() < 2:
        return True
    return False

def is_categorical(spark_df: pyspark.sql.DataFrame, col: str, time_identifier: str, max_unique_numeric_cats: int = 1024, categorical_suffixes: tuple = (), numerical_suffixes: tuple = ()) -> bool:
    """Determine if the given feature should be processed as categorical, as opposed to numeric."""
    if col.endswith(categorical_suffixes):
        if col.endswith(numerical_suffixes):
            print(f"{col} matches both categorical and numerical suffixes; it will be identified as categorical")
        return True
    if isinstance(spark_df.schema[col].dataType, (DateType, TimestampType)):
        return False
    if not isinstance(spark_df.schema[col].dataType, (IntegerType, LongType, ShortType, ByteType, FloatType, DoubleType, DecimalType)):
        if col.endswith(numerical_suffixes):
            print(f"{col} matches numeric suffix but is non-numeric; identified as categorical")
        return True
    if (col.endswith(numerical_suffixes)) or (spark_df.select(col).distinct().count() > max_unique_numeric_cats):
        return False
    return True

def process_single_columns(spark_df, colname, indiv_id = None):
    """Apply data cleaning functions to a singular data column."""
    if colname == indiv_id:
        return spark_df
    if is_degenerate(spark_df, col = colname):
        spark_df = spark_df.drop(colname)
    elif is_categorical(spark_df, colname, indiv_id):
        spark_df = spark_df.withColumn(colname, spark_df[colname].cast(StringType()))
        spark_df = spark_df.fillna('NaN', subset = [colname])
    return spark_df

def process_all_columns(spark_df = pyspark.sql.DataFrame, indiv_id: str = None) -> pyspark.sql.DataFrame:
    """Apply data cleaning functions to all data columns."""
    for col_name in spark_df.columns:
        spark_df = process_single_columns(spark_df, colname = col_name, indiv_id = indiv_id)
    return spark_df

def flag_validation_individuals(spark_df, individual_identifier: 'str', validation_share: float = 0.25) -> pyspark.sql.DataFrame:
    """Flag observations from a random share of individuals."""
    unique_ids = spark_df.select(individual_identifier).distinct()
    total = unique_ids.count()
    val_size = int(validation_share * total)
    unique_ids = unique_ids.sample(True,val_size/total , seed = 9999)
    unique_list = list(unique_ids.select(individual_identifier).toPandas()[individual_identifier])
    spark_df = spark_df.withColumn('val_flagged', spark_df[individual_identifier].isin(unique_list))
    return spark_df

def build_reserved_cols(spark_df: pyspark.sql.DataFrame, test_int: int, individual_identifier: str, time_identifier: str):
    """Add data split and outcome-related columns to the data."""
    spark_df = spark_df.withColumn('_period', spark_df[time_identifier] - 1)
    max_val = lit(spark_df.agg({'_period' : 'max'}).first()[0])
    spark_df = spark_df.withColumn('_period_obs', spark_df['_period'] == max_val)

    max_test_intervals = int((spark_df.select('_period').distinct().count() - 1) / 2)
    test_intervals = test_int
    if test_intervals > max_test_intervals:
        test_intervals = max_test_intervals

    spark_df = spark_df.withColumn('_test', (spark_df['_period'] + test_intervals) >= max_val)

    spark_df = flag_validation_individuals(spark_df, individual_identifier)
    spark_df = spark_df.withColumn('_validation', ~spark_df['_test'] & spark_df['val_flagged'])

    obs_max_window = Window.partitionBy('_test').orderBy(col("_period").desc())
    spark_df = spark_df.withColumn('obs_max_period', F.max(spark_df['_period']).over(obs_max_window))
    spark_df = spark_df.withColumn('_maximum_lead', spark_df.obs_max_period - spark_df['_period'] + (spark_df.obs_max_period < max_val).cast('int'))

    period_window = Window.partitionBy(individual_identifier).orderBy('_period')
    spark_df = spark_df.withColumn("prev_period", lag(spark_df['_period']).over(period_window))
    spark_df = spark_df.withColumn("gaps", spark_df.prev_period < (spark_df['_period'] - 1)).fillna(False)
    spark_df = spark_df.withColumn("gaps", spark_df.gaps.cast('int'))
    spark_df = spark_df.withColumn("_spell", F.sum(spark_df.gaps).over(period_window))
    
    duration_window = Window.partitionBy([individual_identifier, '_spell']).orderBy("_period")
    spark_df = spark_df.withColumn('_duration', F.count(spark_df['_spell']).over(duration_window))
    max_duration_window = Window.partitionBy([individual_identifier, '_spell']).orderBy(col("_period").desc())
    spark_df = spark_df.withColumn('max_duration', F.max(spark_df['_duration']).over(max_duration_window))
    spark_df = spark_df.withColumn('_duration', spark_df.max_duration - spark_df['_duration'])
    spark_df = spark_df.withColumn('_event_observed', spark_df['_duration'] < spark_df['_maximum_lead'])
    
    spark_df = spark_df.drop('max_duration')
    spark_df = spark_df.drop('prev_period')
    spark_df = spark_df.drop('gaps')
    spark_df = spark_df.drop('val_flagged')
    spark_df = spark_df.drop('Obs_max_period')
    return(spark_df)

def sort_panel_data(spark_df: pyspark.sql.DataFrame, individual_identifier: str, time_identifier: str) -> pyspark.sql.DataFrame:
    """Sort the data by individual, then by period."""
    return(spark_df.orderBy(F.asc(individual_identifier), F.asc('_period')))

def build_processed_data(spark_df: pyspark.sql.DataFrame, test_int: int, individual_identifier: str, time_identifier: str, validation_share: float = 0.25):
    """Clean, augment, and store a panel dataset and related information."""
    spark_df.cache()
    check_panel_consistency(spark_df, individual_identifier, time_identifier)
    spark_df = process_all_columns(spark_df, individual_identifier)
    spark_df = build_reserved_cols(spark_df, test_int, individual_identifier, time_identifier)
    spark_df = sort_panel_data(spark_df, individual_identifier, time_identifier)
    return spark_df