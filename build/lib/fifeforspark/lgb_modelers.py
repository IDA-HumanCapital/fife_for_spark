"""FIFE modelers based on LightGBM, which trains gradient-boosted trees."""

from typing import List, Union

import mmlspark.lightgbm.LightGBMClassifier as lgb
import pyspark.sql
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import udf, date_format
from pyspark.sql.types import FloatType
from fifeforspark.base_modelers import default_subset_to_all, Modeler, SurvivalModeler


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
        """
        Train and store a sequence of gradient-boosted tree models.

        Args:
            n_intervals: the maximum periods ahead the model will predict.
            params: Parameters for model tuning

        Returns:
            None
        """
        if n_intervals:
            self.n_intervals = n_intervals
        else:
            self.n_intervals = self.set_n_intervals()
        self.model = self.train(
            params=params
        )

    def train(
            self,
            params: Union[None, dict] = None,
            subset: Union[None, pyspark.sql.column.Column] = None
    ) -> List[pyspark.ml.pipeline.PipelineModel]:
        """
        Train a LightGBM model for each lead length.

        Args:
            params: Parameters for model tuning
            subset: Boolean column for subsetting the data

        Returns:
            List of Pyspark ML Pipeline models
        """
        models = []

        for time_horizon in range(self.n_intervals):
            model = self.train_single_model(
                time_horizon=time_horizon,
                params=params,
                subset=subset
            )
            models.append(model)

        return models

    def train_single_model(
            self,
            time_horizon: int,
            params: Union[None, dict] = None,
            subset: Union[None, pyspark.sql.column.Column] = None
    ) -> pyspark.ml.pipeline.PipelineModel:
        """
        Train a LightGBM model for a single lead length.

        Args:
            time_horizon: The number of periods out for which to build this model
            params: Parameters for model tuning
            subset: Boolean column for subsetting the data

        Returns:
            Single ML Pipeline model
        """
        if params is None:
            params = {
                time_horizon: {
                    "objective": self.objective,
                    "num_iterations": self.config.get("max_epochs", 256),
                }
            }
        params[time_horizon]["num_class"] = self.num_class
        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]
        data = self.label_data(time_horizon)
        data = data.filter(subset)
        data = self.subset_for_training_horizon(data, time_horizon)

        train_data = data.filter(~data[self.validation_col])[
            self.categorical_features + self.numeric_features
            ]
        indexers = [StringIndexer(inputCol=column, outputCol=column + "_index")
                    for column in self.categorical_features]
        feature_columns = [column + "_index" for column in self.categorical_features] + self.numeric_features
        assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
        lgb_model = lgb(featuresCol="features",
                        labelCol="_label",
                        *params[time_horizon],
                        class_weight=data.filter(~data[self.validation_col])[self.weight_col]
                        if self.weight_col
                        else None
                        )
        pipeline = Pipeline(stages=[*indexers, assembler, lgb_model])
        model = pipeline.fit(train_data)
        return model

    def predict(
            self, subset: Union[None, pyspark.sql.column.Column] = None, cumulative: bool = True
    ) -> pyspark.sql.DataFrame:
        """Use trained LightGBM models to predict the outcome for each observation and time horizon.

        Args:
            subset: A Boolean Spark Column that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            cumulative: If True, produce cumulative survival probabilities.
                If False, produce marginal survival probabilities (i.e., one
                minus the hazard rate).

        Returns:
            A dataframe of predictions by observation and lead
            length.
        """
        subset = default_subset_to_all(subset, self.data)
        predict_data = self.data.filter(subset)[self.categorical_features + self.numeric_features]

        firstelement = udf(lambda v: float(v[0]), FloatType())
        first_model = self.model[0]
        predictions = first_model.transform(predict_data).selectExpr('probability as probability_1')
        predictions = predictions.withColumn('probability_1', firstelement(predictions['probability_1']))
        for i, lead_specific_model in enumerate(self.model):
            if i != 0:
                pred_year = lead_specific_model.transform(predict_data).selectExpr(f'probability as probability_{i+1}')

                predictions = predictions.withColumn(f'probability_{i+1}',
                                                     firstelement(pred_year[f'probability_{i+1}']))
                if cumulative:
                    predictions = predictions.withColumn(f'probability_{i + 1}',
                                                         predictions[f'probability_{i + 1}'] *
                                                         predictions[f'probability_{i}'])
        return predictions

    def transform_features(self) -> pyspark.sql.DataFrame:
        """
        Transform datetime features to suit model training.

        Returns:
            Spark DataFrame with transformed features
        """
        data = self.data
        date_cols = [x for x, y in data.dtypes if y in ['date', 'timestamp']]
        for col in date_cols:
            data = data.withColumn(col,
                                   10000*date_format(data[col], "y") +
                                   100*date_format(data[col], "M") +
                                   date_format(data[col], "d"))
        return data

    def save_model(self, file_name: str = "GBT_Model", path: str = "") -> None:
        """
        Save the horizon-specific LightGBM models that comprise the model to disk.
        Functionality does not currently exist

        Args:
            file_name: The desired name of the model on disk
            path: The path for where to save the model

        Returns:
            None
        """
        pass


class LGBSurvivalModeler(LGBModeler, SurvivalModeler):
    """Use LightGBM to forecast probabilities of being observed in future periods."""

    pass