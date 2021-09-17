"""Conduct unit testing for fife.lgb_modelers module."""

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


def test_gbt_predict(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.predict() returns
    survival probabilities for all observations and time intervals.
    """
    errors_list = []
    reserved_cols = [
        "_test",
        "_validation",
        "_predict_obs",
        "_duration",
        "_event_observed",
        "_period",
        "_maximum_lead",
        "_spell",
        "SSNSCR",
    ]
    categorical_features_list = [
                col[0] for col in setup_dataframe.dtypes if col[1] == 'string']
    numeric_features_list = [feature for feature in setup_dataframe.columns
                                     if feature not in (categorical_features_list + reserved_cols)]
    ks_df = ks.DataFrame(setup_dataframe)
    ks_df['FILE_DATE'] = ks_df['FILE_DATE'].factorize(sort = True)[0]
    setup_dataframe = ks_df.to_spark()
    subset_training_obs = (
        ~setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    subset_validation_obs = (
        setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    training_obs_lead_lengths = setup_dataframe.filter(subset_training_obs).groupBy('_duration').count()
    min_survivors_subset = training_obs_lead_lengths[
        training_obs_lead_lengths['_duration'] > setup_config["MIN_SURVIVORS_IN_TRAIN"]
    ]
    n_intervals = min(4, min_survivors_subset.agg({'_duration': 'max'}).first()[0])
    models_list = []
    for time_horizon in np.arange(n_intervals):
        train_data = setup_dataframe[
            (
                setup_dataframe["_duration"] + setup_dataframe["_event_observed"].cast('int')
                > time_horizon
            )
            & subset_training_obs
        ]
        train_data = lgb.Dataset(
            train_data[categorical_features_list + numeric_features_list],
            label=train_data["_duration"] > time_horizon,
        )
        setup_dataframe = setup_dataframe.withColumn('_validation', setup_dataframe.select(
            (
                setup_dataframe["_duration"] + setup_dataframe["_event_observed"]
                > time_horizon
            )
            & subset_validation_obs
        ))
        # validation_data = train_data.create_valid(
        #     validation_data[categorical_features_list + numeric_features_list],
        #     label=validation_data["_duration"] > time_horizon,
        # )
        # model = lgb.train(
        #     {"objective": "binary"},
        #     train_data,
        #     num_boost_round=setup_config["MAX_EPOCHS"],
        #     early_stopping_rounds=setup_config["PATIENCE"],
        #     valid_sets=[validation_data],
        #     valid_names=["validation_set"],
        #     categorical_feature=categorical_features_list,
        #     verbose_eval=True,
        # )
        
        train_data = setup_dataframe.filter(~setup_dataframe['_validation'])[
            categorical_features_list + numeric_features_list + [setup_dataframe['_label']]
            ] 
        train_data = train_data.withColumn('weight', lit(1.0))
        indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").setHandleInvalid("keep")
                    for column in categorical_features_list]
        feature_columns = [column + "_index" for column in categorical_features_list] + numeric_features_list
        assembler = VectorAssembler(inputCols=feature_columns, outputCol='features').setHandleInvalid("keep")
        lgb_model = lgb(featuresCol="features",
                        labelCol="_label",
                        validationIndicatorCol='_validation',
                        weightCol=setup_dataframe.filter(~setup_dataframe['_validation'])['weight']
                        )
        pipeline = Pipeline(stages=[*indexers, assembler, lgb_model])
        model = pipeline.fit(train_data)
        models_list.append(model)
    modeler = GBTSurvivalModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    modeler.model = models_list
    predictions_dict = {}
    subset_bool_series = pd.Series(setup_dataframe["SSNSCR"].mod(2)).astype("bool")
    predictions_dict["cumulative"] = modeler.predict()
    predictions_dict["cumulative_subset"] = modeler.predict(subset=subset_bool_series)
    predictions_dict["marginal"] = modeler.predict(cumulative=False)
    predictions_dict["marginal_subset"] = modeler.predict(
        subset=subset_bool_series, cumulative=False
    )
    for case, predictions in predictions_dict.items():
        if predictions.shape[1] != n_intervals:
            errors_list.append(
                f"Number of periods in predictions array "
                f"({case}) does not match number of "
                f"intervals passed to predict() method."
            )
        if (predictions < 0).any() or (predictions > 1).any():
            errors_list.append(
                f"One or more of the predicted survival "
                f"probabilities ({case}) are outside of "
                f"the range [0, 1]."
            )
        if case in ["cumulative", "marginal"]:
            if predictions.shape[0] != len(setup_dataframe):
                errors_list.append(
                    f"Number of observations in predictions array ({case}) "
                    f"does not match number of observations passed to "
                    f"predict() method."
                )
        if case in ["cumulative_subset", "marginal_subset"]:
            if predictions.shape[0] != subset_bool_series.sum():
                errors_list.append(
                    f"Number of observations in predictions array ({case}) "
                    f"does not match number of observations passed to "
                    f"predict() method."
                )
    if (predictions_dict["cumulative"] > predictions_dict["marginal"]).any():
        errors_list.append(
            "Cumulative probability predictions exceed "
            "marginal probability predictions."
        )
    if (
        predictions_dict["cumulative_subset"] > predictions_dict["marginal_subset"]
    ).any():
        errors_list.append(
            "Cumulative subset probability predictions "
            "exceed marginal subset probability predictions."
        )
    if (
        predictions_dict["cumulative"][:, 0] != predictions_dict["marginal"][:, 0]
    ).any():
        errors_list.append(
            "Cumulative probability predictions do not "
            "match marginal probability predictions for "
            "first time interval."
        )
    if (
        predictions_dict["cumulative_subset"][:, 0]
        != predictions_dict["marginal_subset"][:, 0]
    ).any():
        errors_list.append(
            "Cumulative subset probability predictions do "
            "not match marginal subset probability "
            "predictions for first time interval."
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