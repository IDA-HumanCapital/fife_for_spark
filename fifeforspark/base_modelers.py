from abc import ABC, abstractmethod
from typing import Union, Any
from collections import OrderedDict

import pandas as pd
import numpy as np
import pyspark.pandas as ps
import findspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, when, monotonically_increasing_id, mean, col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score


def default_subset_to_all(
    subset: Union[None, pyspark.sql.DataFrame], data: pyspark.sql.DataFrame
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
        return data.withColumn("True_mask", lit(True)).select("True_mask")
    else:
        assert (
            len(subset.columns) == 1
        ), "Provided subset is not a valid one-column subset"
    return subset


def compute_metrics_for_binary_outcome(
    actuals: pyspark.sql.DataFrame,
    predictions: pyspark.sql.DataFrame,
    total: int,
    threshold_positive: Union[None, str, float] = 0.5,
    share_positive: Union[None, str, float] = None,
    cache=False,
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
    if cache is True:
        predictions.cache()
        actuals.cache()
    actuals = actuals.select(actuals["actuals"].cast(DoubleType()))
    num_true = actuals.agg({"actuals": "sum"}).first()[0]
    metrics = OrderedDict()
    if num_true is None:
        num_true = 0
    if (num_true > 0) & (num_true < total):
        predictions = predictions.withColumn("row_id", monotonically_increasing_id())
        actuals = actuals.withColumn("row_id", monotonically_increasing_id())
        preds_and_labs = predictions.join(
            actuals, predictions.row_id == actuals.row_id
        ).select(predictions.predictions, actuals.actuals)
        if cache is True:
            preds_and_labs.cache()
        preds_and_labs = preds_and_labs.withColumn(
            "rawPrediction", preds_and_labs.predictions.cast(DoubleType())
        )
        evaluator = BinaryClassificationEvaluator(labelCol="actuals")
        metrics["AUROC"] = evaluator.evaluate(
            preds_and_labs, {evaluator.metricName: "areaUnderROC"}
        )
        preds_and_labs = preds_and_labs.drop("rawPrediction")
    else:
        metrics["AUROC"] = np.nan

    # TODO: Add weighted average functionality
    mean_predict = predictions.select(mean(col("predictions"))).first()[0]
    metrics["Actual Share"] = actuals.select(mean(col("actuals"))).first()[0]
    metrics["Predicted Share"] = mean_predict
    if actuals.first() is not None:
        if share_positive == "predicted":
            share_positive = predictions.agg({"predictions": "mean"}).first()[0]
        if share_positive is not None:
            threshold_positive = predictions.approxQuantile(
                "predictions", [1 - share_positive], relativeError=0.1
            )[0]
        elif threshold_positive == "predicted":
            threshold_positive = mean_predict
        preds_and_labs = preds_and_labs.withColumn(
            "predictions",
            when(preds_and_labs.predictions >= threshold_positive, 1).otherwise(0),
        )
        outcomes = {
            "True Positives": [1, 1],
            "False Negatives": [0, 1],
            "False Positives": [1, 0],
            "True Negatives": [0, 0],
        }
        for key in outcomes.keys():
            metrics[key] = (
                preds_and_labs.select(
                    (
                        (preds_and_labs.predictions == outcomes[key][0])
                        & (preds_and_labs.actuals == outcomes[key][1])
                    )
                    .cast("int")
                    .alias("Outcome")
                )
                .agg({"Outcome": "sum"})
                .first()[0]
            )
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
        findspark.init()

        #ks.set_option("compute.ops_on_diff_frames", True)
        ps.config.set_option("compute.ops_on_diff_frames", True)
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
            self.categorical_features = [
                col[0] for col in self.data.dtypes if col[1] == "string"
            ]
            self.numeric_features = [
                feature
                for feature in self.data.columns
                if feature not in (self.categorical_features + self.reserved_cols)
            ]
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
        self, subset: Union[None, pyspark.sql.DataFrame] = None, cumulative: bool = True
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
        subset: Union[None, pyspark.sql.DataFrame] = None,
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
            time_horizon: the number of periods for which you're forecasting (e.g. 2 periods out)

        Returns:
            Spark DataFrame with original dataset subsetted
        """

    @abstractmethod
    def label_data(self, time_horizon: int) -> pyspark.sql.DataFrame:
        """
        Return data with an outcome label for each observation.

        Args:
            time_horizon: the number of periods for which you're forecasting (e.g. 2 periods out)

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
            (
                when(
                    self.data[self.duration_col] <= self.data[self.max_lead_col],
                    self.data[self.duration_col],
                ).otherwise(self.data[self.max_lead_col])
            ).alias("max_durations")
        )
        subset = (
            ~self.data[self.validation_col]
            & ~self.data[self.test_col]
            & ~self.data[self.predict_col]
        )
        train_obs_by_lead_length = train_durations.filter(subset)
        train_obs_by_lead_length = train_obs_by_lead_length.groupBy(
            "max_durations"
        ).count()
        min_survivors_subset = train_obs_by_lead_length.filter(
            (
                train_obs_by_lead_length["count"]
                > self.config.get("MIN_SURVIVOR_IN_TRAIN", 64)
            ).alias("max_durations")
        )
        assert (
            min_survivors_subset.first() is not None
        ), "No lead length has more than 64 survivors."
        n_intervals = min_survivors_subset.agg({"max_durations": "max"}).first()[0]
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
        subset: Union[None, pyspark.sql.DataFrame] = None,
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
        try:
            return self._evaluate_single_node(
                subset=subset,
                threshold_positive=threshold_positive,
                share_positive=share_positive,
            )
        except MemoryError:
            return self._evaluate_multi_node(
                subset=subset,
                threshold_positive=threshold_positive,
                share_positive=share_positive,
            )

    def _evaluate_single_node(
        self,
        subset: Union[None, pyspark.sql.DataFrame] = None,
        threshold_positive: Union[None, str, float] = 0.5,
        share_positive: Union[None, str, float] = None,
    ) -> pd.core.frame.DataFrame:
        """
        Evaluate method for single node. Taken directly from FIFE.

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
            outcome of True, and all elements of the confusion matrix.
        """

        def compute_metrics_for_binary_outcome_single_node(
            actuals: Union[pd.Series, pd.DataFrame],
            predictions: np.ndarray,
            threshold_positive: Union[None, str, float] = 0.5,
            share_positive: Union[None, str, float] = None,
            weights: Union[None, np.ndarray] = None,
        ) -> OrderedDict:
            """Evaluate predicted probabilities against actual binary outcome values.

            Args:
                actuals: A Series representing actual Boolean outcome values.
                predictions: A Series of predicted probabilities of the respective
                    outcome values.
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
                weights: A 1-D array of weights with the same length as actuals and predictions.
                Each prediction contributes to metrics in proportion to its weight. If None, each prediction has the same weight.

            Returns:
                An ordered dictionary containing key-value pairs for area under the
                receiver operating characteristic curve (AUROC), predicted and
                actual shares of observations with an outcome of True, and all
                elements of the confusion matrix.
            """
            metrics = OrderedDict()
            if actuals.any() and not actuals.all():
                metrics["AUROC"] = roc_auc_score(
                    actuals, predictions, sample_weight=weights
                )
            else:
                metrics["AUROC"] = np.nan
            metrics["Predicted Share"] = np.average(predictions, weights=weights)
            metrics["Actual Share"] = np.average(actuals, weights=weights)
            if actuals.empty:
                (
                    metrics["True Positives"],
                    metrics["False Negatives"],
                    metrics["False Positives"],
                    metrics["True Negatives"],
                ) = [0, 0, 0, 0]
            else:
                if share_positive == "predicted":
                    share_positive = metrics["Predicted Share"]
                if share_positive is not None:
                    threshold_positive = np.quantile(predictions, 1 - share_positive)
                elif threshold_positive == "predicted":
                    threshold_positive = metrics["Predicted Share"]
                (
                    metrics["True Positives"],
                    metrics["False Negatives"],
                    metrics["False Positives"],
                    metrics["True Negatives"],
                ) = (
                    confusion_matrix(
                        actuals,
                        predictions >= threshold_positive,
                        labels=[True, False],
                        sample_weight=weights,
                    )
                    .ravel()
                    .tolist()
                )
            return metrics

        if subset is None:
            filtered = self.data.filter(self.data[self.test_col])
            min_val = (
                filtered.select(self.data[self.period_col])
                .agg({self.period_col: "min"})
                .first()[0]
            )
            self.data = self.data.withColumn(
                "subset",
                self.data[self.test_col] & (self.data[self.period_col] == min_val),
            )
            subset = self.data.select("subset")

        predictions = self.predict(subset=subset, cumulative=(not self.allow_gaps))

        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in tqdm(lead_lengths, desc="Evaluating Model by Lead Length"):
            lead_length = int(lead_length)
            labeled_data = self.label_data(int(lead_length - 1))
            if "subset" not in labeled_data.columns:
                labeled_data = labeled_data.to_pandas_on_spark()
                labeled_data["subset"] = subset.to_pandas_on_spark()[list(subset.columns)[0]]
                labeled_data = labeled_data.to_spark()

            actuals = labeled_data.filter(labeled_data["subset"])
            actuals = actuals.filter(actuals[self.max_lead_col] >= lead_length)
            weights = (
                actuals.select(self.weight_col).collect() if self.weight_col else None
            )
            actuals = actuals.select("_label").toPandas()["_label"]
            metrics.append(
                compute_metrics_for_binary_outcome_single_node(
                    actuals,
                    np.array(
                        predictions.select(
                            predictions.columns[lead_length - 1]
                        ).collect()
                    ),
                    threshold_positive=threshold_positive,
                    share_positive=share_positive,
                    weights=weights,
                )
            )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        metrics = metrics.dropna()
        return metrics

    def _evaluate_multi_node(
        self,
        subset: Union[None, pyspark.sql.DataFrame] = None,
        threshold_positive: Union[None, str, float] = 0.5,
        share_positive: Union[None, str, float] = None,
    ) -> pd.core.frame.DataFrame:

        """
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

        predictions = self.predict(subset=subset, cumulative=(not self.allow_gaps))
        lead_lengths = np.arange(self.n_intervals) + 1
        metrics = []
        total = -1
        for lead_length in tqdm(lead_lengths, desc="Evaluating Model by Lead Length"):
            actuals = self.label_data(int(lead_length - 1))
            if subset is None:
                min_val = (
                    actuals.select(actuals["_period"])
                    .agg({"_period": "min"})
                    .first()[0]
                )
                actuals = actuals.withColumn(
                    "subset",
                    actuals[self.test_col] & (actuals[self.period_col] == min_val),
                )
            else:
                actuals = actuals.to_pandas_on_spark()
                actuals["subset"] = subset.to_pandas_on_spark()[list(subset.columns)[0]]
                actuals = actuals.to_spark()

            actuals = actuals.filter(actuals.subset)
            actuals = actuals.filter(actuals[self.max_lead_col] >= int(lead_length))
            actuals = actuals.select(actuals["_label"].alias("actuals"))
            if lead_length == 1:
                total = actuals.count()
            metrics.append(
                compute_metrics_for_binary_outcome(
                    actuals,
                    predictions.select(
                        predictions[int(lead_length - 1)].alias("predictions")
                    ),
                    total=total,
                    threshold_positive=threshold_positive,
                    share_positive=share_positive,
                    cache=self.config.get("CACHE", False),
                )
            )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        metrics = metrics.dropna()
        return metrics

    def forecast(self) -> ps.DataFrame:
        """
        Tabulate survival probabilities for most recent observations.

        Returns:
            Spark DataFrame with forecast results
        """
        columns = [
            str(i + 1) + "-period Survival Probability" for i in range(self.n_intervals)
        ]
        forecasts = self.predict(
            subset=self.data.select(self.predict_col), cumulative=(not self.allow_gaps)
        )
        forecasts = forecasts.to_pandas_on_spark()
        forecasts.columns = columns
        forecast_data = (self.data.filter(self.predict_col)
            .select(self.data[self.config["INDIVIDUAL_IDENTIFIER"]]).alias('copy'))
        forecasts["Index"] = (
            forecast_data.to_pandas_on_spark()
        )
        forecasts = forecasts.set_index("Index")
        # TODO: Add custom index values
        return forecasts

    def subset_for_training_horizon(
        self, data: pyspark.sql.DataFrame, time_horizon: int
    ) -> pyspark.sql.DataFrame:
        """
        Return only observations where the outcome is observed.

        Args:
            data: Dataset to train on
            time_horizon: the number of periods for which you're forecasting (e.g. 2 periods out)

        Returns:
            Spark DataFrame with original dataset subsetted
        """
        if self.allow_gaps:
            return data.filter(data[self.max_lead_col] > time_horizon)
        return data.filter(
            (data[self.duration_col] + data[self.event_col].cast("int")) > time_horizon
        )

    def label_data(self, time_horizon: int) -> pyspark.sql.DataFrame:
        """
        Return data with an outcome label for each observation.

        Args:
            time_horizon: the number of periods for which you're forecasting (e.g. 2 periods out)

        Returns:
            Data with outcome label added based on time_horizon
        """
        spark_df = self.data
        spark_df = spark_df.withColumn(
            self.duration_col,
            when(
                spark_df[self.duration_col] <= spark_df[self.max_lead_col],
                spark_df[self.duration_col],
            ).otherwise(spark_df[self.max_lead_col]),
        )
        if self.allow_gaps:
            ids = spark_df[
                self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]
            ]
            ids = ids.withColumn(
                self.config["INDIVIDUAL_IDENTIFIER"] + "_new",
                ids[self.config["INDIVIDUAL_IDENTIFIER"]],
            )
            ids = ids.withColumn(
                self.config["TIME_IDENTIFIER"] + "_new",
                ids[self.config["TIME_IDENTIFIER"]] - time_horizon - 1,
            )
            ids = ids.withColumn("_label", lit(True))
            ids = ids.drop(
                self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]
            )

            spark_df = spark_df.join(
                ids,
                (
                    spark_df[self.config["INDIVIDUAL_IDENTIFIER"]]
                    == ids[self.config["INDIVIDUAL_IDENTIFIER"] + "_new"]
                )
                & (
                    spark_df[self.config["TIME_IDENTIFIER"]]
                    == ids[self.config["TIME_IDENTIFIER"] + "_new"]
                ),
                "left",
            )
            spark_df = spark_df.drop(
                self.config["INDIVIDUAL_IDENTIFIER"] + "_new",
                self.config["TIME_IDENTIFIER"] + "_new",
            )
            spark_df = spark_df.fillna(False, subset=["_label"])
        else:
            spark_df = spark_df.withColumn(
                "_label", (spark_df[self.duration_col] > lit(time_horizon)).cast("int")
            )
        return spark_df


class StateModeler(Modeler):
    def __init__(self):
        raise NotImplementedError("StateModeler not yet implemented.")


class ExitModeler(StateModeler):
    def __init__(self):
        raise NotImplementedError("ExitModeler not yet implemented.")
