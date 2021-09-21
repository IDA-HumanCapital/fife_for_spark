"""Conduct unit testing for fife.lgb_modelers module."""

from pyspark.ml.classification import GBTClassifier as gbt
from pyspark.ml import PipelineModel, Pipeline
from fifeforspark.gbt_modelers import GBTSurvivalModeler
from pyspark.sql.functions import lit
from pyspark.ml.feature import VectorAssembler, StringIndexer
import lightgbm as lgb
import numpy as np
import pandas as pd
import databricks.koalas as ks


def test_gbt_init():
    """Test that GradientBoostedTreesModeler can be instantiated
    without arguments.
    """
    errors_list = []
    try:
        modeler = GBTSurvivalModeler()
        if not vars(modeler):
            errors_list.append(
                "GradientBoostedTreesModeler instantiated "
                "object does not have attributes."
            )
    except Exception:
        errors_list.append("GradientBoostedTreesModeler could not be instantiated.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))

def test_gbt_train(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.train() returns a list
    containing a trained model for each time interval.
    """
    errors_list = []
    ks_df = ks.DataFrame(setup_dataframe)
    ks_df['FILE_DATE']= ks_df['FILE_DATE'].factorize(sort=True)[0]
    setup_dataframe = ks_df.to_spark()
    subset_training_obs = ~setup_dataframe["_validation"] & ~setup_dataframe["_test"] & ~setup_dataframe["_predict_obs"]
    training_obs_lead_lengths = setup_dataframe.filter(subset_training_obs).groupBy("_duration").count()
    min_survivors_subset = training_obs_lead_lengths.filter(
        training_obs_lead_lengths["_duration"] > setup_config["MIN_SURVIVORS_IN_TRAIN"]
        )
    n_intervals = min(4, min_survivors_subset.agg({'_duration': 'max'}).first()[0])

    modeler = GBTSurvivalModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    modeler.n_intervals = n_intervals

    models_list = modeler.train()
    if len(models_list) != n_intervals:
        errors_list.append(
            "Number of models trained and saved does not "
            "match number of lead lengths."
        )
    for index, model in enumerate(models_list):
        if not isinstance(model, PipelineModel):
            errors_list.append(
                f"Model #{index + 1} (of {len(models_list)}) "
                f"is not an instance of pysparl.ml.PipelineModel."
            )
        print(model)
        if not model.stages[-1].params:
            errors_list.append(
                f"Model #{index + 1} (of {len(models_list)}) "
                f"does not have any training parameters."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbt_train_single_model(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.train_single_model() returns a
    trained model of type lgb.basic.Booster.
    """
    errors_list = []
    ks_df = ks.DataFrame(setup_dataframe)
    ks_df['FILE_DATE'] = ks_df['FILE_DATE'].factorize(sort = True)[0]
    setup_dataframe = ks_df.to_spark()
    subset_training_obs = (
        ~setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    training_obs_lead_lengths = setup_dataframe.filter(subset_training_obs).groupBy('_duration').count()
    min_survivors_subset = training_obs_lead_lengths[
        training_obs_lead_lengths['_duration'] > setup_config["MIN_SURVIVORS_IN_TRAIN"]
    ]
    n_intervals = min(4, min_survivors_subset.agg({'_duration': 'max'}).first()[0])
    modeler = GBTSurvivalModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    for time_horizon in range(n_intervals):
        model = modeler.train_single_model(time_horizon=time_horizon)
        if not isinstance(model, PipelineModel):
            errors_list.append(
                f"Model for time horizon {time_horizon} "
                f"is not an instance of pyspark.ml.PipelineModel."
            )
        if not model.stages[-1].params:
            errors_list.append(
                f"Model for time horizon {time_horizon} "
                f"does not have any training parameters."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbt_build_model(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.build_model()
    creates 'n_intervals' and 'model' attributes.
    """
    errors_list = []
    ks_df = ks.DataFrame(setup_dataframe)
    ks_df['FILE_DATE']= ks_df['FILE_DATE'].factorize(sort=True)[0]
    setup_dataframe = ks_df.to_spark()
    modeler = GBTSurvivalModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    if hasattr(modeler, "n_intervals"):
        errors_list.append(
            'GradientBoostedTreeModeler object has "n_intervals" attribute '
            'before calling "build_model" method.'
        )
    if hasattr(modeler, "model"):
        errors_list.append(
            'GradientBoostedTreeModeler object has "model" attribute '
            'before calling "build_model" method.'
        )
    modeler.build_model(n_intervals = 4, params = {'maxBins': 2000})
    if not hasattr(modeler, "n_intervals"):
        errors_list.append(
            '"Build_model" method does not create "n_intervals" attribute '
            "for GradientBoostedTreeModeler object."
        )
    if not hasattr(modeler, "model"):
        errors_list.append(
            '"Build_model" method does not create "model" attribute '
            "for GradientBoostedTreeModeler object."
        )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbt_set_n_intervals(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.set_n_intervals() returns
    a value within the range of duration values.
    """
    errors_list = []
    min_duration = setup_dataframe.agg({"_duration": "min"}).first()[0]
    max_duration = setup_dataframe.agg({"_duration": "max"}).first()[0]
    modeler = GBTSurvivalModeler(
        config=setup_config, data=setup_dataframe
    )
    n_intervals = modeler.set_n_intervals()
    if not min_duration <= n_intervals <= max_duration:
        errors_list.append(
            'Returned "n_intervals" value is outside the ' "range of duration values."
        )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


SEED = 9999
np.random.seed(SEED)