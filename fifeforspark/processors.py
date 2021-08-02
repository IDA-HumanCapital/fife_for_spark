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

class DataProcessor:
    def __init__(self, config = {}, spark_df = None) -> None:
        if (config.get("individual_identifier", "") == "") and spark_df is not None:
            config["individual_identifier"] = spark_df.columns[0]
        self.config  = config
        self.spark_df = spark_df

    def check_column_consistency(self, colname: str) -> None:
        """Assert column exists, has no missing values, and is not constant."""
        assert colname in self.spark_df.columns, f"{colname} not in data"
        assert self.spark_df.select(isnan(colname).cast('int').alias(colname)).agg({colname:'sum'}).first()[0] == 0, f"{colname} has missing values"
        assert self.spark_df.select(colname).distinct().count() >= 2, f"{colname} does not have multiple unique values"
    
    def is_degenerate(self , col: str) -> bool:
        """Determine if a feature is constant or has too many missing values."""
        if self.spark_df.select(isnan(col).cast('integer').alias(col)).agg({col:'mean'}).first()[0] >= self.config.get('max_null_share', 0.999):
            return True
        if self.spark_df.select(col).distinct().count() < 2:
            return True
        return False

    def is_categorical(self, col: str) -> bool:
        """Determine if the given feature should be processed as categorical, as opposed to numeric."""
        if col.endswith(self.config.get('categorical_suffixes', ())):
            if col.endswith(self.config.get('numerical_suffixes', ())):
                print(f"{col} matches both categorical and numerical suffixes; it will be identified as categorical")
            return True
        if isinstance(self.spark_df.schema[col].dataType, (DateType, TimestampType)):
            return False
        if not isinstance(self.spark_df.schema[col].dataType, (IntegerType, LongType, ShortType, ByteType, FloatType, DoubleType, DecimalType)):
            if col.endswith(self.config.get('numerical_suffixes', ())):
                print(f"{col} matches numeric suffix but is non-numeric; identified as categorical")
            return True
        if (col.endswith(self.config.get('numerical_suffixes', ()))) or (self.spark_df.select(col).distinct().count() > self.config.get('max_unique_categories', 1024)):
            return False
        return True
    
class PanelDataProcessor(DataProcessor):
    def __init__(self, config, spark_df) -> None:
        if (config.get("time_identifier", "") == "") and spark_df is not None:
            config["time_identifier"] = spark_df.columns[1]
        super().__init__(config, spark_df)
        
    def check_panel_consistency(self) -> None:
        """Ensure observations have unique individual-period combinations."""
        self.check_column_consistency(self.config['individual_identifier'])
        self.check_column_consistency(self.config['time_identifier'])
        subset = self.spark_df.select([self.config['individual_identifier'], self.config['time_identifier']])
        assert subset.count() == subset.dropDuplicates().count(), "One or more individuals have multiple observations for a single time value"

    def process_single_columns(self, colname):
        """Apply data cleaning functions to a singular data column."""
        if colname == self.config['individual_identifier']:
            return self.spark_df
        if self.is_degenerate(col = colname):
            self.spark_df = self.spark_df.drop(colname)
        elif self.is_categorical(colname):
            self.spark_df = self.spark_df.withColumn(colname, self.spark_df[colname].cast(StringType()))
            self.spark_df = self.spark_df.fillna('NaN', subset = [colname])
        return self.spark_df

    def process_all_columns(self, indiv_id: str = None) -> pyspark.sql.DataFrame:
        """Apply data cleaning functions to all data columns."""
        for col_name in self.spark_df.columns:
            self.spark_df = self.process_single_columns(colname = col_name)
        return self.spark_df

    def flag_validation_individuals(self, individual_identifier: 'str', validation_share: float = 0.25) -> pyspark.sql.DataFrame:
        """Flag observations from a random share of individuals."""
        unique_ids = self.spark_df.select(self.config['individual_identifier']).distinct()
        total = unique_ids.count()
        val_size = int(self.config.get('validation_share', 0.25) * total)
        unique_ids = unique_ids.sample(True,val_size/total , seed = 9999)
        unique_list = list(unique_ids.select(self.config['individual_identifier']).toPandas()[self.config['individual_identifier']])
        self.spark_df = self.spark_df.withColumn('val_flagged', self.spark_df[self.config['individual_identifier']].isin(unique_list))
        return self.spark_df

    def build_reserved_cols(self):
        """Add data split and outcome-related columns to the data."""
        self.spark_df = self.spark_df.withColumn('_period', self.spark_df[self.config['time_identifier']] - 1)
        max_val = lit(self.spark_df.agg({'_period' : 'max'}).first()[0])
        self.spark_df = self.spark_df.withColumn('_period_obs', self.spark_df['_period'] == max_val)

        max_test_intervals = int((self.spark_df.select('_period').distinct().count() - 1) / 2)
        if self.config.get('test_intervals', 0) > max_test_intervals:
            self.config['test_intervals'] = max_test_intervals

        self.spark_df = self.spark_df.withColumn('_test', (self.spark_df['_period'] + self.config.get('test_intervals', 0)) >= max_val)

        self.spark_df = self.flag_validation_individuals(self.spark_df, self.config['individual_identifier'])
        self.spark_df = self.spark_df.withColumn('_validation', ~self.spark_df['_test'] & self.spark_df['val_flagged'])

        obs_max_window = Window.partitionBy('_test').orderBy(col("_period").desc())
        self.spark_df = self.spark_df.withColumn('obs_max_period', F.max(self.spark_df['_period']).over(obs_max_window))
        self.spark_df = self.spark_df.withColumn('_maximum_lead', self.spark_df.obs_max_period - self.spark_df['_period'] + (self.spark_df.obs_max_period < max_val).cast('int'))

        period_window = Window.partitionBy(self.config['individual_identifier']).orderBy('_period')
        self.spark_df = self.spark_df.withColumn("prev_period", lag(self.spark_df['_period']).over(period_window))
        self.spark_df = self.spark_df.withColumn("gaps", self.spark_df.prev_period < (self.spark_df['_period'] - 1)).fillna(False)
        self.spark_df = self.spark_df.withColumn("gaps", self.spark_df.gaps.cast('int'))
        self.spark_df = self.spark_df.withColumn("_spell", F.sum(self.spark_df.gaps).over(period_window))

        duration_window = Window.partitionBy([self.config['individual_identifier'], '_spell']).orderBy("_period")
        self.spark_df = self.spark_df.withColumn('_duration', F.count(self.spark_df['_spell']).over(duration_window))
        max_duration_window = Window.partitionBy([self.config['individual_identifier'], '_spell']).orderBy(col("_period").desc())
        self.spark_df = self.spark_df.withColumn('max_duration', F.max(self.spark_df['_duration']).over(max_duration_window))
        self.spark_df = self.spark_df.withColumn('_duration', self.spark_df.max_duration - self.spark_df['_duration'])
        self.spark_df = self.spark_df.withColumn('_event_observed', self.spark_df['_duration'] < self.spark_df['_maximum_lead'])

        self.spark_df = self.spark_df.drop('max_duration')
        self.spark_df = self.spark_df.drop('prev_period')
        self.spark_df = self.spark_df.drop('gaps')
        self.spark_df = self.spark_df.drop('val_flagged')
        self.spark_df = self.spark_df.drop('Obs_max_period')
        return(self.spark_df)

    def sort_panel_data(self) -> pyspark.sql.DataFrame:
        """Sort the data by individual, then by period."""
        return(self.spark_df.orderBy(F.asc(self.config['individual_identifier']), F.asc('_period')))

    def build_processed_data(self):
        """Clean, augment, and store a panel dataset and related information."""
        self.spark_df.cache()
        self.check_panel_consistency()
        self.spark_df = self.process_all_columns()
        self.spark_df = self.build_reserved_cols()
        self.spark_df = self.sort_panel_data()
        return self.spark_df