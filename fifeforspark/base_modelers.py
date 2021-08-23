from abc import ABC, abstractmethod
from typing import Union, Any
from collections import OrderedDict

import pandas as pd
import numpy as np
import databricks.koalas as ks
import findspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, when, monotonically_increasing_id
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from lifelines.utils import concordance_index

findspark.init()

spark = SparkSession.builder.getOrCreate()


def default_subset_to_all(    
    subset: Union[None, pd.core.series.Series], data: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Map an unspecified subset to an entirely True boolean mask.

    Args:
        subset: the subset passed into the data (boolean column) if it exists
        data: the original dataset

    Returns:
        Original subset argument if not none, otherwise a new boolean mask that is always True.
    """

    if subset is None:
        return data.withColumn('True_mask', lit(True)).select('True_mask')
    return subset


def compute_metrics_for_binary_outcomes(
    actuals: pyspark.sql.DataFrame, predictions: pyspark.sql.DataFrame,
    threshold_positive: Union[None, str, float] = 0.5, share_positive: Union[None, str, float] = None
) -> OrderedDict:
    """
    Function to compute performance metrics for binary classification models

    Args:
        actuals: The actual values
        predictions: The predicted values
        threshold_positive: The threshold for determining whether a value is predicted True or False
        share_positive: The share of values that you want to classify as positive

    Returns:
        Ordered dictionary with evaluation metrics
    """
    actuals = actuals.select(actuals['actuals'].cast(DoubleType()))
    num_true = actuals.agg({'actuals': 'sum'}).first()[0]
    total = actuals.count()
    metrics = OrderedDict()
    if (num_true > 0) & (num_true < total):
        predictions = predictions.withColumn(
            "row_id", monotonically_increasing_id())
        actuals = actuals.withColumn("row_id", monotonically_increasing_id())
        preds_and_labs = predictions.join(actuals, predictions.row_id == actuals.row_id).select(
            predictions.predictions, actuals.actuals)
        scoresAndLabels = preds_and_labs.rdd.map(tuple)
        evaluator = BinaryClassificationMetrics(scoresAndLabels)
        metrics['AUROC'] = evaluator.areaUnderROC

    else:
        metrics["AUROC"] = np.nan

    # Difficult to do a weighted avg in pyspark, leaving out for now
    mean_predict = predictions.agg({'predictions': 'mean'}).first()[0]
    mean_actual = actuals.agg({'actuals': 'mean'}).first()[0]
    metrics["Predicted Share"] = mean_predict
    metrics["Actual Share"] = mean_actual
    if actuals.first() is not None:
        if share_positive == 'predicted':
            share_positive = predictions.agg(
                {'predictions': 'mean'}).first()[0]
        if share_positive is not None:
            threshold_positive = predictions.approxQuantile(
                'predictions', [1-share_positive], relativeError=.1)[0]
        elif threshold_positive == 'predicted':
            threshold_positive = mean_predict
        
        preds_and_labs = preds_and_labs.withColumn('predictions', when(
            preds_and_labs.predictions >= threshold_positive, 1).otherwise(0))
        
        outcomes = {'True Positives': [1,1], 'False Negatives': [0,1], 'False Positives': [1,0], 'True Negatives': [0,0]}
        for key in outcomes.keys():
            metrics[key] = preds_and_labs.select(((preds_and_labs.predictions == outcomes[key][0]) & (
                preds_and_labs.actuals == outcomes[key][1])).cast('int').alias('Outcome')).agg({'Outcome': 'sum'}).first()[0]

    return metrics


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
        data: Union[None, pyspark.sql.DataFrame] = None,
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
        if (config.get("time_identifier", "") == "") and data is not None:
            config["time_identifier"] = data.columns[1]

        if (config.get("individual_identifier", "") == "") and data is not None:
            config["individual_identifier"] = data.columns[0]

        findspark.init()
        self.spark = SparkSession.builder.getOrCreate()
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
            self.reserved_cols.append(self.config["individual_identifier"])
        if self.weight_col:
            self.reserved_cols.append(self.weight_col)
        if self.data is not None:
            self.categorical_features = [
                col[0] for col in self.data.dtypes if col[1] == 'string']
            self.numeric_features = [feature for feature in self.data.columns
                                     if feature not in (self.categorical_features + self.reserved_cols)]
            self.data = self.transform_features()

    @abstractmethod
    def train(self) -> Any:
        """
        Train and return a model.

        Returns:
            Any
        """

    @abstractmethod
    def predict(
        self, subset: Union[None, pyspark.sql.column.Column] = None, cumulative: bool = True
    ) -> pyspark.sql.DataFrame:
        """
        Use trained model to produce observation survival probabilities.

        Args:
            subset: The subset of values to predict on
            cumulative: Flag for whether output probabilities should be cumulative

        Returns:
            Spark dataframe with prediction results
        """

    @abstractmethod
    def evaluate(
        self,
        subset: Union[None, pyspark.sql.column.Column] = None,
        threshold_positive: Union[None, str, float] = 0.5,
        share_positive: Union[None, str, float] = None,
    ) -> pd.core.frame.DataFrame:
        """
        Tabulate model performance metrics.

        Args:
            subset: The subset of values to evaluate
            threshold_positive: The threshold value for determining whether a value is positive or negative
            share_positive: The share you want to classify as positive

        Returns:
            Pandas Dataframe with evaluation results
        """

    @abstractmethod
    def forecast(self) -> pyspark.sql.DataFrame:
        """
        Tabulate survival probabilities for most recent observations.

        Returns:
            Spark DataFrame with forecast results
        """

    @abstractmethod
    def subset_for_training_horizon(
        self, data: pyspark.sql.DataFrame, time_horizon: int
    ) -> pyspark.sql.DataFrame:
        """
        Return only observations where the outcome is observed.

        Args:
            data: Dataset to train on
            time_horizon: the number of periods for which you're forecasting (i.e. 2 periods out)

        Returns:
            Spark DataFrame with original dataset subsetted
        """

    @abstractmethod
    def label_data(self, time_horizon: int) -> pyspark.sql.DataFrame:
        """
        Return data with an outcome label for each observation.

        Args:
            time_horizon: the number of periods for which you're forecasting (i.e. 2 periods out)

        Returns:
            Data with outcome label added based on time_horizon
        """

    @abstractmethod
    def save_model(self, path: str = "") -> None:
        """
        Save model file(s) to disk.

        Args:
            path: Path to save model

        Returns:
            None
        """

    @abstractmethod
    def transform_features(self) -> pyspark.sql.DataFrame:
        """
        Transform datetime features to suit model training.

        Returns:
            Spark DataFrame with transformed features
        """

    @abstractmethod
    def build_model(self, n_intervals: Union[None, int] = None) -> None:
        """
        Configure, train, and store a model.

        Args:
            n_intervals: the maximum periods ahead the model will predict.

        Returns:
            None
        """

    def set_n_intervals(self) -> int:
        """
        Determine the maximum periods ahead the model will predict.

        Returns:
            n_intervals (the maximum periods ahead the model will predict)
        """
        train_durations = self.data.select(
            (when(
                self.data[self.duration_col] <= self.data[self.max_lead_col], self.data[self.duration_col]
            ).otherwise(self.data[self.max_lead_col])).alias('max_durations')
        )
        subset = ~self.data[self.validation_col] & ~self.data[self.test_col] & ~self.data[self.predict_col]
        train_obs_by_lead_length = train_durations.filter(subset)
        train_obs_by_lead_length = train_obs_by_lead_length.groupBy(
            'max_durations').count()
        min_survivors_subset = train_obs_by_lead_length.filter(
            (train_obs_by_lead_length['count'] > self.config.get(
                "min_survivors_in_train", 64)).alias('max_durations'))
        assert min_survivors_subset.first() is not None, "No lead length has more than 64 survivors."
        n_intervals = min_survivors_subset.agg(
            {'max_durations': 'max'}
        ).first()[0]
        return n_intervals


class SurvivalModeler(Modeler):
    """Forecast probabilities of being observed in future periods.

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
        allow_gaps (bool): Whether or not observations should be included for
            training and evaluation if there is a period without an observation
            between the period of observations and the last period of the
            given time horizon.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SurvivalModeler.

        Args:
            **kwargs: Arguments to Modeler.__init__().
        """
        super().__init__(**kwargs)
        self.objective = "binary"
        self.num_class = 1

    def evaluate(
        self,
        subset: Union[None, pyspark.sql.column.Column] = None,
        threshold_positive: Union[None, str, float] = 0.5,
        share_positive: Union[None, str, float] = None,
    ) -> pd.core.frame.DataFrame:
        """
        Tabulate model performance metrics.

        Args:
            subset: A Boolean Series that is True for observations over which
                the metrics will be computed. If None, default to all test
                observations in the earliest period of the test set.
            threshold_positive: None, "predicted", or a value in [0, 1]
                representing the minimum predicted probability considered to be a
                positive prediction; Specify "predicted" to use the predicted share
                positive in each time horizon. Overridden by share_positive.
            share_positive: None, "predicted", or a value in [0, 1] representing
                the share of observations with the highest predicted probabilities
                considered to have positive predictions. Specify "predicted" to use
                the predicted share positive in each time horizon. Probability ties
                may cause share of positives in output to exceed given value.
                Overrides threshold_positive.
        Returns:
            A DataFrame containing, for the binary outcomes of survival to each
            lead length, area under the receiver operating characteristic
            curve (AUROC), predicted and actual shares of observations with an
            outcome of True, and all elements of the confusion matrix. Also
            includes concordance index over the restricted mean survival time.
        """
        filtered = self.data.filter(self.data[self.test_col])
        min_val = filtered.select(self.data[self.period_col]).agg(
            {self.period_col: 'min'}).first()[0]
        if subset is None:
            subset = self.data[self.test_col] & self.data.select(
                self.data[self.period_col] == min_val)
        predictions = self.predict(
            subset=subset, cumulative=(not self.allow_gaps))
        lead_lengths = np.arange(self.n_intervals) + 1
        metrics = []
        for lead_length in lead_lengths:
            actuals = self.label_data(lead_length - 1).filter(subset)
            actuals = actuals.filter(actuals[self.max_lead_col] >= lead_length)
            actuals = actuals.select(actuals["_label"])
            metrics.append(
                compute_metrics_for_binary_outcomes(
                    actuals,
                    predictions.select(
                        predictions[lead_length - 1]).limit(actuals.count()),
                    threshold_positive=threshold_positive,
                    share_positive=share_positive,
                )
            )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        metrics["Other Metrics:"] = ""
        if (not self.allow_gaps) and (self.weight_col is None):
            concordance_index_value = concordance_index(
                self.data[subset][[self.duration_col,
                                   self.max_lead_col]].min(axis=1),
                np.sum(predictions, axis=-1),
                self.data[subset][self.event_col],
            )
            metrics["C-Index"] = np.where(
                metrics.index == 1, concordance_index_value, ""
            )
        metrics = metrics.dropna()
        return metrics

    def forecast(self) -> ks.DataFrame:
        """
        Tabulate survival probabilities for most recent observations.

        Returns:
            Spark DataFrame with forecast results
        """
        columns = [
            str(i + 1) + "-period Survival Probability" for i in range(self.n_intervals)]
        forecasts = self.predict(
            subset=self.data[self.predict_col], cumulative=(not self.allow_gaps))
        forecasts = forecasts.to_koalas()
        index = self.data.filter(self.data[self.predict_col]).select(
            self.data[self.config["individual_identifier"]])
        forecasts.columns = columns
        forecasts['index'] = index
        return forecasts

    def subset_for_training_horizon(self, data: pyspark.sql.DataFrame, time_horizon: int) -> pyspark.sql.DataFrame:
        """
        Return only observations where the outcome is observed.

        Args:
            data: Dataset to train on
            time_horizon: the number of periods for which you're forecasting (i.e. 2 periods out)

        Returns:
            Spark DataFrame with original dataset subsetted
        """
        if self.allow_gaps:
            return data.filter(data[self.max_lead_col] > time_horizon)
        return data.filter((data[self.duration_col] + data[self.event_col]).cast('int') > time_horizon)

    def label_data(self, time_horizon: int) -> pyspark.sql.DataFrame:
        """
        Return data with an outcome label for each observation.

        Args:
            time_horizon: the number of periods for which you're forecasting (i.e. 2 periods out)

        Returns:
            Data with outcome label added based on time_horizon
        """
        # Spark automatically creates a copy when setting one value equal to another, different from python
        spark_df = self.data
        spark_df = spark_df.withColumn(spark_df[self.duration_col], when(
            spark_df[self.duration_col] <= spark_df[self.max_lead_col], spark_df[self.duration_col]).otherwise(spark_df[self.max_lead_col]))
        if self.allow_gaps:
            ids = spark_df[self.config["individual_identifier"],
                           self.config["time_identifer"]]
            ids = ids.withColumn(
                self.config["individual_identifier"] + '_new', ids[self.config["individual_identifier"]])
            ids = ids.withColumn(
                self.config["time_identifier"] + '_new', ids[self.config["time_identifer"]] - time_horizon - 1)
            ids = ids.withColumn('_label', lit(True))
            ids = ids.drop(
                self.config["individual_identifier"], self.config["time_identifier"])

            spark_df = spark_df.join(ids, (spark_df[self.config["individual_identifier"]] == ids[self.config["individual_identifier"]]) & (
                spark_df[self.config["time_identifier"]] == ids[self.config["time_identifier"]]), "left")
            spark_df = spark_df.drop(
                self.config["individual_identifier"] + '_new', self.config["time_identifier"] + '_new')
            spark_df = spark_df.fillna(False, subset=['_label'])
        else:
            spark_df.withColumn(
                '_label', spark_df[self.duration_col] > lit(time_horizon))
        return spark_df


class StateModeler(Modeler):
    pass


class ExitModeler(StateModeler):
    pass
