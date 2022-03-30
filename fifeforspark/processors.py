import findspark
import pyspark
import pyspark.sql
import pyspark.sql.functions as F
import pyspark.pandas as ps
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import isnan, lit, col, lag
from pyspark.sql.types import (
    DateType,
    TimestampType,
    StringType,
    IntegerType,
    LongType,
    ShortType,
    ByteType,
    FloatType,
    DoubleType,
    DecimalType,
)
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
        if (config.get("INDIVIDUAL_IDENTIFIER", "") == "") and data is not None:
            config["INDIVIDUAL_IDENTIFIER"] = data.columns[0]
            print(
                "Individual identifier column name not given; assumed to be "
                f'leftmost column ({config["INDIVIDUAL_IDENTIFIER"]})'
            )
        self.config = config
        self.data = data

    def check_column_consistency(self, colname: str) -> None:
        """
        Assert column exists, has no missing values, and is not constant.

        Args:
            colname: The name of the column to check

        Returns:
            None
        """
        if colname == self.config["TIME_IDENTIFIER"]:
            ps_df = ps.DataFrame(self.data)
            ps_df["_period"] = ps_df[self.config["TIME_IDENTIFIER"]].factorize(
                sort=True
            )[0]
            self.data = ps_df.to_spark()
            colname = "_period"
        assert colname in self.data.columns, f"{colname} not in data"
        assert (
            self.data.select(isnan(colname).cast("int").alias(colname))
            .agg({colname: "sum"})
            .first()[0]
            == 0
        ), f"{colname} has missing values"
        assert (
            self.data.select(colname).distinct().count() >= 2
        ), f"{colname} does not have multiple unique values"

    def is_degenerate(self, colname: str) -> bool:
        """
        Determine if a feature is constant or has too many missing values

        Args:
            col: The column/feature to check

        Returns:
            Boolean value for whether the column is degenerate
        """
        if colname == self.config["TIME_IDENTIFIER"]:
            ps_df = ps.DataFrame(self.data)
            ps_df["_period"] = ps_df[self.config["TIME_IDENTIFIER"]].factorize(
                sort=True
            )[0]
            self.data = ps_df.to_spark()
            colname = "_period"
        if self.data.select(isnan(colname).cast("integer").alias(colname)).agg(
            {colname: "mean"}
        ).first()[0] >= self.config.get("MAX_NULL_SHARE", 0.999):
            return True
        if self.data.select(colname).distinct().count() < 2:
            return True
        return False

    def is_categorical(self, col: str) -> bool:
        """
        Determine if the given feature should be processed as categorical, as opposed to numeric.

        Args:
            col: The column to check

        Returns:
            Boolean value for whether the column is categorical
        """
        try:
            if col.endswith(self.config.get("CATEGORICAL_SUFFIXES", ())):
                if col.endswith(self.config.get("NUMERIC_SUFFIXES", ())):
                    print(
                        f"{col} matches both categorical and numerical suffixes; it will be identified as categorical"
                    )
                return True
        except TypeError:
            raise TypeError(
                "'NUMERIC_SUFFIXES' and 'CATEGORICAL_SUFFIXES' must be either strings or tuples of strings."
            )
        if isinstance(self.data.schema[col].dataType, (DateType, TimestampType)):
            return False
        if not isinstance(
            self.data.schema[col].dataType,
            (
                IntegerType,
                LongType,
                ShortType,
                ByteType,
                FloatType,
                DoubleType,
                DecimalType,
            ),
        ):
            if col.endswith(self.config.get("NUMERIC_SUFFIXES", ())):
                print(
                    f"{col} matches numeric suffix but is non-numeric; identified as categorical"
                )
            return True
        if (col.endswith(self.config.get("NUMERIC_SUFFIXES", ()))) or (
            self.data.select(col).distinct().count()
            > self.config.get("MAX_UNIQUE_CATEGORIES", 1024)
        ):
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

    def __init__(
        self,
        config: Union[None, dict] = {},
        data: Union[None, pyspark.sql.DataFrame] = None,
        shuffle_parts=200,
    ) -> None:
        """Initialize the PanelDataProcessor.

        Args:
            config: A dictionary of configuration parameters.
            data: A DataFrame to be processed.
        """

        if (config.get("TIME_IDENTIFIER", "") == "") and data is not None:
            config["TIME_IDENTIFIER"] = data.columns[1]
            print(
                "Time identifier column name not given; assumed to be "
                f'second-leftmost column ({config["TIME_IDENTIFIER"]})'
            )
        super().__init__(config, data)
        self.spark.conf.set("spark.sql.shuffle.partitions", shuffle_parts)

    def check_panel_consistency(self) -> None:
        """
        Ensure observations have unique individual-period combinations.

        Returns:
            None
        """
        self.check_column_consistency(self.config["INDIVIDUAL_IDENTIFIER"])
        self.check_column_consistency(self.config["TIME_IDENTIFIER"])
        subset = self.data.select(
            [self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]]
        )
        assert (
            subset.count() == subset.dropDuplicates().count()
        ), "One or more individuals have multiple observations for a single time value."

    def process_single_column(self, colname) -> pyspark.sql.DataFrame:
        """
        Apply data cleaning functions to a singular data column.

        Args:
            colname: The column to process

        Returns:
            Dataframe: dataframe with the processed column
        """
        if colname == self.config["INDIVIDUAL_IDENTIFIER"]:
            return self.data
        if self.is_degenerate(colname=colname):
            self.data = self.data.drop(colname)
        elif self.is_categorical(colname):
            self.data = self.data.withColumn(
                colname, self.data[colname].cast(StringType())
            )
            self.data = self.data.fillna("NaN", subset=[colname])
        return self.data

    def process_all_columns(self) -> pyspark.sql.DataFrame:
        """
        Apply data cleaning functions to all data columns.

        Returns:
            Spark DataFrame with processed columns
        """
        for col_name in self.data.columns:
            self.data = self.process_single_column(colname=col_name)
        return self.data

    def flag_validation_individuals(self) -> pyspark.sql.DataFrame:
        """
        Flag observations from a random share of individuals.

        Returns:
            Spark DataFrame with flagged validation individuals
        """
        unique_ids = self.data.select(self.config["INDIVIDUAL_IDENTIFIER"]).distinct()
        total = unique_ids.count()
        val_size = int(self.config.get("VALIDATION_SHARE", 0.25) * total)
        unique_ids = unique_ids.sample(True, val_size / total, seed=9999)
        unique_list = list(
            unique_ids.select(self.config["INDIVIDUAL_IDENTIFIER"]).toPandas()[
                self.config["INDIVIDUAL_IDENTIFIER"]
            ]
        )
        self.data = self.data.withColumn(
            "val_flagged",
            self.data[self.config["INDIVIDUAL_IDENTIFIER"]].isin(unique_list),
        )
        return self.data

    def build_reserved_cols(self) -> pyspark.sql.DataFrame:
        """
        Add data split and outcome-related columns to the data.

        Returns:
            Spark DataFrame with reserved columns added
        """

        max_val = lit(self.data.agg({"_period": "max"}).first()[0])
        self.data = self.data.withColumn(
            "_predict_obs", self.data["_period"] == max_val
        )

        max_test_intervals = int(
            (self.data.select("_period").distinct().count() - 1) / 2
        )
        if self.config.get("TEST_INTERVALS", 0) > max_test_intervals:
            warn(
                f"The specified value for TEST_INTERVALS was too high and will not allow for enough training periods. It was automatically reduced to {max_test_intervals}"
            )
            self.config["TEST_INTERVALS"] = max_test_intervals

        self.data = self.data.withColumn(
            "_test",
            (self.data["_period"] + self.config.get("TEST_INTERVALS", 0)) >= max_val,
        )

        self.data = self.flag_validation_individuals()
        self.data = self.data.withColumn(
            "_validation", ~self.data["_test"] & self.data["val_flagged"]
        )

        obs_max_window = Window.partitionBy("_test").orderBy(col("_period").desc())
        self.data = self.data.withColumn(
            "obs_max_period", F.max(self.data["_period"]).over(obs_max_window)
        )
        self.data = self.data.withColumn(
            "_maximum_lead",
            self.data.obs_max_period
            - self.data["_period"]
            + (self.data.obs_max_period < max_val).cast("int"),
        )

        period_window = Window.partitionBy(
            self.config["INDIVIDUAL_IDENTIFIER"]
        ).orderBy("_period")
        self.data = (
            self.data.withColumn(
                "gaps",
                lag(self.data["_period"]).over(period_window)
                < (self.data["_period"] - 1),
            )
            .fillna(False)
            .withColumn("gaps", col("gaps").cast("int"))
        )

        self.data = self.data.withColumn(
            "_spell", F.sum(self.data.gaps).over(period_window)
        )

        duration_window = Window.partitionBy(
            [self.config["INDIVIDUAL_IDENTIFIER"], "_spell"]
        ).orderBy("_period")
        self.data = self.data.withColumn(
            "_duration", F.count(self.data["_spell"]).over(duration_window)
        )

        max_duration_window = Window.partitionBy(
            [self.config["INDIVIDUAL_IDENTIFIER"], "_spell"]
        ).orderBy(col("_period").desc())
        self.data = self.data.withColumn(
            "_duration",
            F.max(self.data["_duration"]).over(max_duration_window)
            - self.data["_duration"],
        )
        self.data = self.data.withColumn(
            "_event_observed", self.data["_duration"] < self.data["_maximum_lead"]
        )

        self.data = self.data.drop("gaps")
        self.data = self.data.drop("val_flagged")
        self.data = self.data.drop("obs_max_period")
        return self.data

    def sort_panel_data(self) -> pyspark.sql.DataFrame:
        """
        Sort the data by individual, then by period.

        Returns:
            Sorted panel data
        """

        return self.data.orderBy(
            F.asc(self.config["INDIVIDUAL_IDENTIFIER"]), F.asc("_period")
        )

    def build_processed_data(self):
        """
        Clean, augment, and store a panel dataset and related information.

        Returns:
            Processed data
        """

        if self.config.get("CACHE", False) == True:
            self.data.cache()
        self.check_panel_consistency()
        self.data = self.process_all_columns()
        self.data = self.build_reserved_cols()
        self.data = self.sort_panel_data()
        return self.data
