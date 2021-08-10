import findspark
import pyspark
import pyspark.sql
import pyspark.sql.functions as F
import databricks.koalas as ks
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import isnan, lit, col, lag
from pyspark.sql.types import DateType, TimestampType, StringType, IntegerType, LongType, ShortType, ByteType, FloatType, DoubleType, DecimalType
from typing import Union
from warnings import warn


class DataProcessor:
    """Prepare data by identifying features as degenerate or categorical."""

    def __init__(self, config={}, data=None) -> None:
        """Initialize the DataProcessor.
        Args:
            config: A dictionary of configuration parameters.
            data: A DataFrame to be processed.
        """
        findspark.init()
        self.spark = SparkSession.builder.getOrCreate()
        if (config.get("individual_identifier", "") == "") and data is not None:
            config["individual_identifier"] = data.columns[0]
            print(
                "Individual identifier column name not given; assumed to be "
                f'leftmost column ({config["individual_identifier"]})'
            )
        self.config = config
        self.data = data

    def check_column_consistency(self, colname: str) -> None:
        """Assert column exists, has no missing values, and is not constant."""
        assert colname in self.data.columns, f"{colname} not in data"
        assert self.data.select(
            isnan(colname).cast('int').alias(colname)
        ).agg({colname: 'sum'}).first()[0] == 0, f"{colname} has missing values"
        assert self.data.select(
            colname).distinct().count() >= 2, f"{colname} does not have multiple unique values"

    def is_degenerate(self, col: str) -> bool:
        """Determine if a feature is constant or has too many missing values."""
        if self.data.select(
                isnan(col).cast('integer').alias(col)
        ).agg({col: 'mean'}).first()[0] >= self.config.get('max_null_share', 0.999):
            return True
        if self.data.select(col).distinct().count() < 2:
            return True
        return False

    def is_categorical(self, col: str) -> bool:
        """Determine if the given feature should be processed as categorical, as opposed to numeric."""
        if col.endswith(self.config.get('categorical_suffixes', ())):
            if col.endswith(self.config.get('numerical_suffixes', ())):
                print(
                    f"{col} matches both categorical and numerical suffixes; it will be identified as categorical"
                )
            return True
        if isinstance(self.data.schema[col].dataType, (DateType, TimestampType)):
            return False
        if not isinstance(self.data.schema[col].dataType, (IntegerType, LongType, ShortType,
                                                           ByteType, FloatType, DoubleType, DecimalType)):
            if col.endswith(self.config.get('numerical_suffixes', ())):
                print(
                    f"{col} matches numeric suffix but is non-numeric; identified as categorical"
                )
            return True
        if isinstance(self.data.schema[col].dataType, (IntegerType, LongType, ShortType,
                                                       ByteType, FloatType, DoubleType, DecimalType)):
            return False
        if (col.endswith(self.config.get('numerical_suffixes', ()))) or (
                self.data.select(col).distinct().count() > self.config.get('max_unique_categories', 1024)):
            return False
        return True


class PanelDataProcessor(DataProcessor):
    """Ready panel data for modelling.

    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): Processed panel data.
        raw_subset (pd.core.frame.DataFrame): An unprocessed sample from the
            final period of data. Useful for displaying meaningful values in
            SHAP plots.
        categorical_maps (dict): Contains for each categorical feature a map
            from each unique value to a whole number.
        numeric_ranges (pd.core.frame.DataFrame): Contains for each numeric
            feature the maximum and minimum value in the training set.
    """

    def __init__(self, config: Union[None, dict] = {},
                 data: Union[None, pyspark.sql.DataFrame] = None,) -> None:
        """Initialize the PanelDataProcessor.
        Args:
            config: A dictionary of configuration parameters.
            data: A DataFrame to be processed.
        """

        if (config.get("time_identifier", "") == "") and data is not None:
            config["time_identifier"] = data.columns[1]
            print(
                "Time identifier column name not given; assumed to be "
                f'second-leftmost column ({config["time_identifier"]})'
            )
        super().__init__(config, data)

    def check_panel_consistency(self) -> None:
        """Ensure observations have unique individual-period combinations."""
        self.check_column_consistency(self.config['individual_identifier'])
        self.check_column_consistency(self.config['time_identifier'])
        subset = self.data.select(
            [self.config['individual_identifier'], self.config['time_identifier']])
        assert subset.count() == subset.dropDuplicates().count(
        ), "One or more individuals have multiple observations for a single time value."

    def process_single_column(self, colname):
        """Apply data cleaning functions to a singular data column."""
        if colname == self.config['individual_identifier']:
            return self.data
        if self.is_degenerate(col=colname):
            self.data = self.data.drop(colname)
        elif self.is_categorical(colname):
            self.data = self.data.withColumn(
                colname, self.data[colname].cast(StringType()))
            self.data = self.data.fillna('NaN', subset=[colname])
        return self.data

    def process_all_columns(self, indiv_id: str = None) -> pyspark.sql.DataFrame:
        """Apply data cleaning functions to all data columns."""
        for col_name in self.data.columns:
            self.data = self.process_single_column(colname=col_name)
        return self.data

    def flag_validation_individuals(
            self, individual_identifier: 'str', validation_share: float = 0.25
    ) -> pyspark.sql.DataFrame:
        """Flag observations from a random share of individuals."""
        unique_ids = self.data.select(
            self.config['individual_identifier']).distinct()
        total = unique_ids.count()
        val_size = int(self.config.get('validation_share', 0.25) * total)
        unique_ids = unique_ids.sample(True, val_size/total, seed=9999)
        unique_list = list(
            unique_ids.select(self.config['individual_identifier']
                              ).toPandas()[self.config['individual_identifier']])
        self.data = self.data.withColumn(
            'val_flagged', self.data[self.config['individual_identifier']].isin(unique_list))
        return self.data

    def build_reserved_cols(self):
        """Add data split and outcome-related columns to the data."""
        ks_df = ks.DataFrame(self.data)
        ks_df['_period'] = ks_df[self.config['time_identifier']].factorize(sort=True)[
            0]
        self.data = ks_df.to_spark()

        max_val = lit(self.data.agg({'_period': 'max'}).first()[0])
        self.data = self.data.withColumn(
            '_period_obs', self.data['_period'] == max_val)

        max_test_intervals = int(
            (self.data.select('_period').distinct().count() - 1) / 2)
        if self.config.get('test_intervals', 0) > max_test_intervals:
            warn(
                f"The specified value for TEST_INTERVALS was too high and will not allow for enough training periods. It was automatically reduced to {max_test_intervals}"
            )
            self.config['test_intervals'] = max_test_intervals

        self.data = self.data.withColumn(
            '_test', (self.data['_period'] + self.config.get('test_intervals', 0)) >= max_val)

        self.data = self.flag_validation_individuals(
            self.data, self.config['individual_identifier'])
        self.data = self.data.withColumn(
            '_validation', ~self.data['_test'] & self.data['val_flagged'])

        obs_max_window = Window.partitionBy(
            '_test').orderBy(col("_period").desc())
        self.data = self.data.withColumn('obs_max_period', F.max(
            self.data['_period']).over(obs_max_window))
        self.data = self.data.withColumn(
            '_maximum_lead', self.data.obs_max_period - self.data['_period'] + (
                self.data.obs_max_period < max_val).cast('int'))

        period_window = Window.partitionBy(
            self.config['individual_identifier']).orderBy('_period')
        self.data = self.data.withColumn("prev_period", lag(
            self.data['_period']).over(period_window))
        self.data = self.data.withColumn(
            "gaps", self.data.prev_period < (self.data['_period'] - 1)).fillna(False)
        self.data = self.data.withColumn("gaps", self.data.gaps.cast('int'))
        self.data = self.data.withColumn(
            "_spell", F.sum(self.data.gaps).over(period_window))

        duration_window = Window.partitionBy(
            [self.config['individual_identifier'], '_spell']).orderBy("_period")
        self.data = self.data.withColumn('_duration', F.count(
            self.data['_spell']).over(duration_window))

        max_duration_window = Window.partitionBy(
            [self.config['individual_identifier'], '_spell']).orderBy(col("_period").desc())
        self.data = self.data.withColumn('max_duration', F.max(
            self.data['_duration']).over(max_duration_window))
        self.data = self.data.withColumn(
            '_duration', self.data.max_duration - self.data['_duration'])
        self.data = self.data.withColumn(
            '_event_observed', self.data['_duration'] < self.data['_maximum_lead'])

        self.data = self.data.drop('max_duration')
        self.data = self.data.drop('prev_period')
        self.data = self.data.drop('gaps')
        self.data = self.data.drop('val_flagged')
        self.data = self.data.drop('Obs_max_period')
        return(self.data)

    def sort_panel_data(self) -> pyspark.sql.DataFrame:
        """Sort the data by individual, then by period."""
        return(self.data.orderBy(F.asc(self.config['individual_identifier']), F.asc('_period')))

    def build_processed_data(self):
        """Clean, augment, and store a panel dataset and related information."""
        self.data.cache()
        self.check_panel_consistency()
        self.data = self.process_all_columns()
        self.data = self.build_reserved_cols()
        self.data = self.sort_panel_data()
        return self.data
