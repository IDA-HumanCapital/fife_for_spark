from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import lit, when
import findspark
import pyspark
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from abc import ABC
from typing import Union
import databricks.koalas as ks

spark = SparkSession.builder.getOrCreate()

def compute_metrics_for_binary_outcomes(actuals, predictions, threshold_positive, share_positive) -> pyspark.sql.DataFrame:
    #If any values in the actuals are true, and not all values are true:
    #Assume that both actuals and predictions are Spark DFs
    actuals = actuals.select(actuals['actuals'].cast(DoubleType()))
    num_true = actuals.agg({'actuals':'sum'}).first()[0]
    total = actuals.count()
    schema = StructType([
            StructField('AUROC', FloatType(), True),
            StructField('Predicted Share', FloatType(), True),
            StructField('Actual Share', FloatType(), True),
            StructField('True Positives', IntegerType(), True),
            StructField('False Negatives', IntegerType(), True),
            StructField('False Positives', IntegerType(), True),
            StructField('True Negatives', IntegerType(), True)])
    metrics = spark.createDataFrame([(.0,.0,.0,0,0,0,0)], schema = schema)

    if (num_true > 0) & (num_true < total):
        w = Window.orderBy('predictions')
        predictions = predictions.withColumn("row_id",row_number().over(w))

        w = Window.orderBy('actuals')
        actuals = actuals.withColumn("row_id",row_number().over(w))
        
        preds_and_labs = predictions.join(actuals, predictions.row_id == actuals.row_id).select(predictions.predictions, actuals.actuals)
        scoresAndLabels = preds_and_labs.rdd.map(tuple)
        evaluator = BinaryClassificationMetrics(scoresAndLabels)
        metrics = metrics.withColumn('AUROC', lit(evaluator.areaUnderROC))
    
    else:
        metrics = metrics.withColumn('AUROC', lit(np.nan))

    #Difficult to do a weighted avg in pyspark
    mean_predict = predictions.agg({'predictions':'mean'}).first()[0]
    mean_actual = actuals.agg({'actuals':'mean'}).first()[0]
    metrics = metrics.withColumn('Predicted Share', lit(mean_predict))
    metrics = metrics.withColumn('Actual Share', lit(mean_actual))

    # Checks if actuals has a single row / if not then it's empty
    if actuals.first() == None:
        #Values are already set to zero by default
        pass
    
    else:
        if share_positive == 'predicted':
            share_positive = lit(predictions.agg({'predictions':'mean'}).first()[0])
        #if share_positive is not None:
            # SPARKIFY THIS
        #    threshold_positive = np.quantile(predictions, 1 - share_positive)
        elif threshold_positive == 'predicted': 
            threshold_positive = mean_predict
            
        preds_and_labs.withColumn('predictions', when(preds_and_labs.predictions >= threshold_positive, 1).otherwise(0))
        TP = preds_and_labs.select(((preds_and_labs.predictions == 1) & (preds_and_labs.actuals == 1)).cast('int').alias('TP')).agg({'TP':'sum'}).first()[0]
        TN = preds_and_labs.select(((preds_and_labs.predictions == 0) & (preds_and_labs.actuals == 0)).cast('int').alias('TN')).agg({'TN':'sum'}).first()[0]
        FP = preds_and_labs.select(((preds_and_labs.predictions == 1) & (preds_and_labs.actuals == 0)).cast('int').alias('FP')).agg({'FP':'sum'}).first()[0]
        FN = preds_and_labs.select(((preds_and_labs.predictions == 0) & (preds_and_labs.actuals == 1)).cast('int').alias('FN')).agg({'FN':'sum'}).first()[0]
        
        metrics = metrics.withColumn('True Positive', lit(TP))
        metrics = metrics.withColumn('False Negative', lit(FN))
        metrics = metrics.withColumn('False Positive', lit(FP))
        metrics = metrics.withColumn('True Negative', lit(TN))
        
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
        """Initialize the SurvivalModeler.
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
    ) -> ks.frame.DataFrame:
        """Tabulate model performance metrics.
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
        min_val = filtered.select(self.data[self.period_col]).agg({self.period_col:'min'}).first()[0]
        if subset is None:
            subset = self.data[self.test_col] & self.data.select(self.data[self.period_col] == lit(min_val))
        predictions = self.predict(subset=subset, cumulative=(not self.allow_gaps))
        schema = StructType([
                    StructField('AUROC', FloatType(), True),
                    StructField('Predicted Share', FloatType(), True),
                    StructField('Actual Share', FloatType(), True),
                    StructField('True Positives', IntegerType(), True),
                    StructField('False Negatives', IntegerType(), True),
                    StructField('False Positives', IntegerType(), True),
                    StructField('True Negatives', IntegerType(), True)])
        metrics = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD, schema = schema)
    
        lead_lengths = np.arange(self.n_intervals) + 1
    
        for lead_length in lead_lengths:
            actuals = self.label_data(lead_length - 1).filter(subset)
            actuals = actuals.filter(actuals[self.max_lead_col] >= lit(lead_length))
            weights = actuals.select(actuals[self.weight_col]) if self.weight_col in self.data.columns else None
            actuals = actuals.select(actuals["_label"])
            metrics.union(
                compute_metrics_for_binary_outcome(
                    actuals,
                    predictions[:, lead_length - 1][actuals.index],
                    threshold_positive=threshold_positive,
                    share_positive=share_positive,
                    weights=weights,
                )
            )
        #Pyspark dataframes don't have an index
        #Pyspark doesn't have built-in concordance_index
        #Maybe we can use koalas here? I believe it'll solve both problems, but at the cost of some performance
        metrics = metrics.to_koalas
        metrics['Lead Length'] = lead_lengths
        metrics.set_index('Lead Length')
        metrics["Other Metrics:"] = ""
        #TO DO: code parallelized c-index or convert pyspark dfs into lists
        metrics = metrics.dropna()
        return metrics
    
    def forecast(self) -> pyspark.sql.DataFrame:
        columns = [str(i + 1) + "-period Survival Probability" for i in range(self.n_intervals)]
        forecasts = self.predict(subset=self.data[self.predict_col], cumulative=(not self.allow_gaps))
        forecasts = forecasts.to_koalas()
        index = self.data.filter(self.data[self.predict_col]).select(self.data[self.config["individual_identifier"]])
        forecasts.columns = columns
        forecasts.set_index = index
        return forecasts
    
    def subset_for_training_horizon(self, data: pyspark.sql.DataFrame, time_horizon: int) -> pyspark.sql.DataFrame:
        """Return only observations where survival would be observed."""
        if self.allow_gaps:
            return data.select(data[self.max_lead_col] > lit(time_horizon))
        return data.select((data[self.duration_col] + data[self.event_col]).cast('int') > lit(time_horizon))
    
    def label_data(self, time_horizon: int, duration_col: str, max_lead_col: str) -> pyspark.sql.DataFrame:
        """Return data with an indicator for survival for each observation."""
        #Spark automatically creates a copy when setting one value equal to another, different from python
        spark_df = self
        spark_df = spark_df.withColumn(duration_col, when(duration_col <= max_lead_col, duration_col).otherwise(max_lead_col))
        if self.allow_gaps:
            ids = spark_df[self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]]
            ids = ids.withColumn(self.config["INDIVIDUAL_IDENTIFIER"] + '_new', ids[self.config["INDIVIDUAL_IDENTIFIER"]])
            ids = ids.withColumn(self.config["TIME_IDENTIFIER"] + '_new', ids[self.config["TIME_IDENTIFIER"]] - time_horizon - 1)
            ids = ids.withColumn('_label', lit(True))
            ids = ids.drop(self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"])
            
            spark_df = spark_df.join(ids,(spark_df[self.config["INDIVIDUAL_IDENTIFIER"]] == ids[self.config["INDIVIDUAL_IDENTIFIER"]]) & (spark_df[self.config["TIME_IDENTIFIER"]] == ids[self.config["TIME_IDENTIFIER"]]),"left")
            spark_df = spark_df.drop(self.config["INDIVIDUAL_IDENTIFIER"] + '_new', self.config["TIME_IDENTIFIER"] + '_new')
            spark_df = spark_df.fillna(False, subset = ['_label'])
        else:
            spark_df.withColumn('_label', spark_df[self.duration_col] > lit(time_horizon))

class StateModeler(Modeler):
    pass

class ExitModeler(StateModeler):
    pass