"""FIFE modelers based on Pyspark RFC, which trains gradient-boosted trees."""

from typing import Union

import pyspark.sql
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from fifeforspark.base_modelers import SurvivalModeler
from fifeforspark.lgb_modelers import LGBModeler
from pyspark.ml.classification import RandomForestClassifier as rfc
from pyspark.sql.functions import lit
from warnings import warn
import pyspark.pandas as ps


class RFCModeler(LGBModeler):
    def train_single_model(
        self,
        time_horizon: int,
        params: Union[None, dict] = None,
        subset: Union[None, pyspark.sql.DataFrame] = None,
    ) -> pyspark.ml.pipeline.PipelineModel:
        """
        Train a Random Forest model for a single lead length.
        Args:
            time_horizon: The number of periods out for which to build this model
            params: Parameters for model tuning
            subset: Boolean column for subsetting the data
        Returns:
            Single ML Pipeline model
        """
        warn(
            "Current functionality does not support any missing values when using in DataBricks. Please remove any "
            "missing values or label them as their own category. If you're not using this package in DataBricks, "
            "current functionality does support missing values. Further investigation into this issue is underway."
        )

        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]

        else:
            self.data = self.data.to_pandas_on_spark()
            self.data["subset"] = subset.to_pandas_on_spark()[list(subset.columns)[0]]
            self.data = self.data.to_spark()
            subset = self.data["subset"]
            self.data = self.data.drop("subset")

        data = self.label_data(time_horizon)
        data = data.filter(subset)
        data = self.subset_for_training_horizon(data, time_horizon)

        train_data = data.filter(~data[self.validation_col])[
            self.categorical_features + self.numeric_features + [data["_label"]]
        ]

        weight_col = self.weight_col
        if not self.weight_col:
            train_data = train_data.withColumn("weight", lit(1.0))
            weight_col = "weight"

        indexers = [
            StringIndexer(
                inputCol=column, outputCol=column + "_index"
            ).setHandleInvalid("keep")
            for column in self.categorical_features
        ]
        feature_columns = [
            column + "_index" for column in self.categorical_features
        ] + [x for x in train_data.columns if x in self.numeric_features]
        assembler = VectorAssembler(
            inputCols=feature_columns, outputCol="features"
        ).setHandleInvalid("keep")

        if params is None:
            rfc_model = rfc(
                featuresCol="features", labelCol="_label", weightCol=weight_col
            )
        else:
            rfc_model = rfc(
                featuresCol="features",
                labelCol="_label",
                **params,
                weightCol=weight_col
            )

        pipeline = Pipeline(stages=[*indexers, assembler, rfc_model])
        model = pipeline.fit(train_data)
        return model


class RFCSurvivalModeler(RFCModeler, SurvivalModeler):
    """Use Pyspark's ML RFC Classifier to forecast probabilities of being observed in future periods."""

    pass
