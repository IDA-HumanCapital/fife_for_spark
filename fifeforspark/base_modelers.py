from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, when
import findspark
import pyspark
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from abc import ABC
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

class Modeler(ABC):
    """Set template for modelers that use panel data to produce forecasts.
    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): User-provided panel data.
        categorical_features (list): Column names of categorical features.
        duration_col (str): Name of the column representing the number of
            future periods observed for the given individual.
        event_col (str): Name of the column indicating whether the individual
            is observed to exit the dataset.
        predict_col (str): Name of the column indicating whether the
            observation will be used for prediction after training.
        test_col (str): Name of the column indicating whether the observation
            will be used for testing model performance after training.
        validation_col (str): Name of the column indicating whether the
            observation will be used for evaluating model performance during
            training.
        period_col (str): Name of the column representing the number of
            periods since the earliest period in the data.
        max_lead_col (str): Name of the column representing the number of
            observable future periods.
        spell_col (str): Name of the column representing the number of
            previous spells of consecutive observations of the same individual.
        weight_col (str): Name of the column representing observation weights.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of periods ahead to forecast
        allow_gaps (bool): Whether or not observations should be included for
            training and evaluation if there is a period without an observation
            between the period of observations and the last period of the
            given time horizon.
    """

    def __init__(
        self,
        config: Union[None, dict] = {},
        data: Union[None, pd.core.frame.DataFrame] = None,
        duration_col: str = "_duration",
        event_col: str = "_event_observed",
        predict_col: str = "_predict_obs",
        test_col: str = "_test",
        validation_col: str = "_validation",
        period_col: str = "_period",
        max_lead_col: str = "_maximum_lead",
        spell_col: str = "_spell",
        weight_col: Union[None, str] = None,
        allow_gaps: bool = False,
    ) -> None:
        """Characterize data for modelling.
        Identify each column as categorical, reserved, and numeric.
        Compute the maximum lead length for modelling.
        Args:
            config: User-provided configuration parameters.
            data: User-provided panel data.
            duration_col: Name of the column representing the number of future
                periods observed for the given individual.
            event_col: Name of the column indicating whether the individual is
                observed to exit the dataset.
            predict_col: Name of the column indicating whether the observation
                will be used for prediction after training.
            test_col: Name of the column indicating whether the observation
                will be used for testing model performance after training.
            validation_col: Name of the column indicating whether the
                observation will be used for evaluating model performance
                during training.
            period_col (str): Name of the column representing the number of
                periods since the earliest period in the data.
            max_lead_col (str): Name of the column representing the number of
                observable future periods.
            spell_col (str): Name of the column representing the number of
                previous spells of consecutive observations of the same individual.
            weight_col (str): Name of the column representing observation weights.
            allow_gaps (bool): Whether or not observations should be included for
                training and evaluation if there is a period without an observation
                between the period of observations and the last period of the
                given time horizon.
        """
        if (config.get("TIME_IDENTIFIER", "") == "") and data is not None:
            config["TIME_IDENTIFIER"] = data.columns[1]

        if (config.get("INDIVIDUAL_IDENTIFIER", "") == "") and data is not None:
            config["INDIVIDUAL_IDENTIFIER"] = data.columns[0]
            
        self.config = config
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col
        self.predict_col = predict_col
        self.test_col = test_col
        self.validation_col = validation_col
        self.period_col = period_col
        self.max_lead_col = max_lead_col
        self.spell_col = spell_col
        self.weight_col = weight_col
        self.allow_gaps = allow_gaps
        self.reserved_cols = [
            self.duration_col,
            self.event_col,
            self.predict_col,
            self.test_col,
            self.validation_col,
            self.period_col,
            self.max_lead_col,
            self.spell_col,
        ]
        if self.config:
            self.reserved_cols.append(self.config["INDIVIDUAL_IDENTIFIER"])
        if self.weight_col:
            self.reserved_cols.append(self.weight_col)
        if self.data is not None:
            self.categorical_features = [col[0] for col in self.data.dtypes if col[1] == 'string']
            self.numeric_features = [feature for feature in self.data.columns
                if feature not in (self.categorical_features + self.reserved_cols)]
            self.data = self.transform_features()
                 
    def set_n_intervals(self) -> int:
        """Determine the maximum periods ahead the model will predict."""
        train_durations = self.data.select(
            when(
                self.duration_col <= self.max_lead_col, self.duration_col
                ).otherwise(self.max_lead_col).alias('max_durations')
        )
        subset = ~self.data[self.validation_col] & ~self.data[self.test_col] & ~self.data[self.predict_col]
        train_obs_by_lead_length = train_durations.filter(subset)
        train_obs_by_lead_length.groupBy('max_durations').count()
        n_intervals = train_obs_by_lead_length.select(
            train_obs_by_lead_length.max_durations > self.config.get("MIN_SURVIVORS_IN_TRAIN", 64).alias('max_durations')
        ).agg(
            {'max_durations':'max'}
        ).first()[0]
        return n_intervals

class SurvivalModeler(Modeler):
    pass

class StateModeler(Modeler):
    pass

class ExitModeler(StateModeler):
    pass