"""FIFE modelers based on LightGBM, which trains gradient-boosted trees."""

import json
from typing import List, Union
from warnings import warn

from fifeforspark.base_modelers import default_subset_to_all, Modeler, SurvivalModeler

import findspark
import pyspark
import pyspark.sql
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import isnan, lit, col, lag
from pyspark.sql.types import DateType, TimestampType, StringType, IntegerType, LongType, ShortType, ByteType, FloatType, DoubleType, DecimalType
import pyspark.sql.functions as F

import mmlspark.lightgbm.LightGBMClassifier as lgb

import pandas as pd
import numpy as np


class LGBModeler(Modeler):
    """Train a gradient-boosted tree model for each lead length using MMLSpark's LightGBM.
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
        n_intervals (int): The largest number of periods ahead to forecast.
        model (list): A trained LightGBM model (lgb.basic.Booster) for each
            lead length.
        objective (str): The LightGBM model objective appropriate for the
            outcome type, which is "binary" for binary classification.
        num_class (int): The num_class LightGBM parameter, which is 1 for
            binary classification.
    """

    def build_model(
        self,
        n_intervals: Union[None, int] = None,
        params: dict = None
    ) -> None:
        """Train and store a sequence of gradient-boosted tree models."""
        if n_intervals:
            self.n_intervals = n_intervals
        else:
            self.n_intervals = self.set_n_intervals()
        early_stopping = (params is None) or (
            any(["num_iterations" not in d for d in params.values()])
        )
        self.model = self.train(
            params=params,
            validation_early_stopping=early_stopping
        )

    def train(
        self,
        params: Union[None, dict] = None,
        subset: Union[None, pd.core.series.Series] = None,
        validation_early_stopping: bool = True,
    ) -> List[lgb.basic.Booster]:
        """Train a LightGBM model for each lead length."""
        models = []

        for time_horizon in range(self.n_intervals):
            model = self.train_single_model(
                time_horizon=time_horizon,
                params=params,
                subset=subset,
                validation_early_stopping=validation_early_stopping,
            )
            models.append(model)

        return models

    def train_single_model(
        self,
        time_horizon: int,
        params: Union[None, dict] = None,
        subset: Union[None, pd.core.series.Series] = None,
        validation_early_stopping: bool = True,
    ) -> lgb.basic.Booster:
        """Train a LightGBM model for a single lead length."""
        if params is None:
            params = {
                time_horizon: {
                    "objective": self.objective,
                    "num_iterations": self.config.get("MAX_EPOCHS", 256),
                }
            }
        params[time_horizon]["num_class"] = self.num_class
        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]
        data = self.label_data(time_horizon)
        data = data[subset]
        data = self.subset_for_training_horizon(data, time_horizon)
        if validation_early_stopping:
            train_data = lgb.Dataset(
                data[~data[self.validation_col]][
                    self.categorical_features + self.numeric_features
                ],
                label=data[~data[self.validation_col]]["_label"],
                weight=data[~data[self.validation_col]][self.weight_col]
                if self.weight_col
                else None,
            )
            validation_data = train_data.create_valid(
                data[data[self.validation_col]][
                    self.categorical_features + self.numeric_features
                ],
                label=data[data[self.validation_col]]["_label"],
                weight=data[data[self.validation_col]][self.weight_col]
                if self.weight_col
                else None,
            )
            model = lgb.train(
                params[time_horizon],
                train_data,
                early_stopping_rounds=self.config.get("PATIENCE", 4),
                valid_sets=[validation_data],
                valid_names=["validation_set"],
                categorical_feature=self.categorical_features,
                verbose_eval=True,
            )
        else:
            data = lgb.Dataset(
                data[self.categorical_features + self.numeric_features],
                label=data["_label"],
                weight=data[self.weight_col] if self.weight_col else None,
            )
            model = lgb.train(
                params[time_horizon],
                data,
                categorical_feature=self.categorical_features,
                verbose_eval=True,
            )
        return model

    def predict(
        self, subset: Union[None, pd.core.series.Series] = None, cumulative: bool = True
    ) -> np.ndarray:
        """Use trained LightGBM models to predict the outcome for each observation and time horizon.
        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            cumulative: If True, produce cumulative survival probabilies.
                If False, produce marginal survival probabilities (i.e., one
                minus the hazard rate).
        Returns:
            A numpy array of predictions by observation and lead
            length.
        """
        subset = default_subset_to_all(subset, self.data)
        predict_data = self.data[self.categorical_features + self.numeric_features][
            subset
        ]
        predictions = np.array(
            [
                lead_specific_model.predict(predict_data)
                for lead_specific_model in self.model
            ]
        ).T
        if cumulative:
            predictions = np.cumprod(predictions, axis=1)
        return predictions

    def transform_features(self) -> pd.DataFrame:
        """Transform features to suit model training."""
        data = self.data.copy(deep=True)
        if self.config.get("DATETIME_AS_DATE", True):
            date_cols = list(data.select_dtypes("datetime").columns) + [
                col
                for col in data.select_dtypes("category")
                if np.issubdtype(data[col].cat.categories.dtype, np.datetime64)
            ]
            for col in date_cols:
                data[col] = (
                    data[col].dt.year * 10000
                    + data[col].dt.month * 100
                    + data[col].dt.day
                )
        else:
            data[data.select_dtypes("datetime").columns] = data[
                data.select_dtypes("datetime").columns
            ].apply(pd.to_numeric)
            for col in data.select_dtypes("category"):
                if np.issubdtype(data[col].cat.categories.dtype, np.datetime64):
                    data[col] = data[col].astype(int)
        return data

    def save_model(self, file_name: str = "GBT_Model", path: str = "") -> None:
        """Save the horizon-specific LightGBM models that comprise the model to disk."""
        for i, lead_specific_model in enumerate(self.model):
            with open(f"{path}{i + 1}-lead_{file_name}.json", "w") as file:
                json.dump(lead_specific_model.dump_model(), file, indent=4)


class LGBSurvivalModeler(LGBModeler, SurvivalModeler):
    """Use LightGBM to forecast probabilities of being observed in future periods."""

    pass

