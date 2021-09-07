"""FIFE modelers based on Pyspark GBT, which trains gradient-boosted trees."""
from typing import List, Union

import pyspark.sql
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from fifeforspark.base_modelers import SurvivalModeler
from fifeforspark.lgb_modelers import LGBModeler
from pyspark.ml.classification import GBTClassifier as gbt


class GBTModeler(LGBModeler):

    def train_single_model(
            self,
            time_horizon: int,
            params: Union[None, dict] = None,
            subset: Union[None, pyspark.sql.DataFrame] = None
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

        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]

        else:
            self.data = self.data.to_koalas()
            self.data['subset'] = subset.to_koalas()[list(subset.columns)[0]]
            self.data = self.data.to_spark()
            subset = self.data['subset']
            self.data = self.data.drop('subset')

        data = self.label_data(time_horizon)
        data = data.filter(subset)
        data = self.subset_for_training_horizon(data, time_horizon)

        train_data = data.filter(~data[self.validation_col])[
            self.categorical_features + self.numeric_features + [data['_label']]
            ]
        indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").setHandleInvalid("keep")
                    for column in self.categorical_features]
        feature_columns = [column + "_index" for column in self.categorical_features] + self.numeric_features
        feature_columns = [x for x in train_data.columns if x in feature_columns] # reset the order
        assembler = VectorAssembler(inputCols=feature_columns, outputCol='features').setHandleInvalid("keep")
        max_bins = max([32]+[len(train_data[feature].unique()) for feature in self.categorical_features])

        if params is None:
            gbt_model = gbt(featuresCol="features",
                            labelCol="_label",
                            maxIter=5,
                            maxBins=max_bins,
                            weightCol=data.filter(~data[self.validation_col])[self.weight_col]
                            if self.weight_col
                            else None
                            )
        else:
            gbt_model = gbt(featuresCol="features",
                            labelCol="_label",
                            **params,
                            maxIter=5,
                            maxBins=max_bins,
                            weightCol=data.filter(~data[self.validation_col])[self.weight_col]
                            if self.weight_col
                            else None
                            )

        pipeline = Pipeline(stages=[*indexers, assembler, gbt_model])
        model = pipeline.fit(train_data)
        return model


class GBTSurvivalModeler(GBTModeler, SurvivalModeler):
    """Use Pyspark's ML GBT Classifier to forecast probabilities of being observed in future periods."""

    pass