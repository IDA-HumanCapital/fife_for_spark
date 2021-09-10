"""Conduct unit testing for fifeforspark.processors module."""
import sys, os
sys.path.append(os.path.abspath("C:/Users/jwang/Documents/GitHub/fife_for_spark/fifeforspark"))
import processors
import numpy as np
import pandas as pd
import pytest
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import rand
from pyspark.sql.types import StringType, DoubleType
findspark.init()
spark = SparkSession.builder.getOrCreate()

def test_DataProcessor(setup_config, setup_dataframe):
    """Test that DataProcessor binds config and data arguments."""
    data_processor = processors.DataProcessor(setup_config, setup_dataframe)
    assert (data_processor.config == setup_config) and (data_processor.data ==
        setup_dataframe)

def test_is_degenerate(setup_config, setup_dataframe):
    """Test that is_degenerate identifies an invariant feature as degenerate."""
    setup_dataframe.show()
    data_processor = processors.DataProcessor(config=setup_config, data=setup_dataframe)
    assert data_processor.is_degenerate("constant_categorical_var")

def test_is_categorical(setup_config, setup_dataframe):
    """Test that is_categorical correctly identifies
    categorical and numeric features.
    """
    reserved_cat_cols = [
        "_duration",
        "_event_observed",
        "_predict_obs",
        "_test",
        "_validation",
        "_period",
        "_maximum_lead",
    ]
    reserved_num_cols = ["FILE_DATE"]
    non_reserved_cols = [
        x
        for x in setup_dataframe.columns
        if x not in reserved_cat_cols + reserved_num_cols
    ]
    setup_cat_cols = [x for x in non_reserved_cols if "categorical" in x] + [
        'completely_null_var'
        ]
    setup_num_cols = [x for x in non_reserved_cols if "numeric" in x]
    num_vars_treated_as_cat_vars = []
    for num_col in setup_num_cols:
        if (
            setup_dataframe.select(num_col).distinct().count()
            <= setup_config["MAX_UNIQUE_CATEGORIES"]
        ):
            num_vars_treated_as_cat_vars.append(num_col)
    setup_cat_cols = setup_cat_cols + num_vars_treated_as_cat_vars + reserved_cat_cols
    setup_num_cols = [
        x for x in setup_num_cols if x not in num_vars_treated_as_cat_vars
    ] + reserved_num_cols
    data_processor = processors.DataProcessor(config=setup_config, data=setup_dataframe)
    for col in setup_cat_cols:
        assert data_processor.is_categorical(col)
    for col in setup_num_cols:
        assert ~data_processor.is_categorical(col)
        
def test_PanelDataProcessor(setup_config, setup_dataframe):
    """Test that PanelDataProcessor binds config and data arguments."""
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    assert (data_processor.config == setup_config) and (data_processor.data ==
        setup_dataframe
    )
    
def test_process_single_column(setup_config, setup_dataframe):
    """Test that PanelDataProcessor.process_single_column() drops degenerate
    columns, correctly casts categorical columns, and does not modify individual
    identifier column."""
    errors_list = []
    indiv_id_col = setup_config["INDIVIDUAL_IDENTIFIER"]
    degenerate_cols = [
        col
        for col in setup_dataframe.columns
        if (setup_dataframe.dropna(subset = [col]).first() is None) | (setup_dataframe.select(col).distinct().count() < 2)
    ]
    categorical_cols = [
        col
        for col in setup_dataframe.columns
        if ("categorical_var" in col) & (col not in degenerate_cols)
    ]
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    processed_col = data_processor.process_single_column(indiv_id_col).select(indiv_id_col)
    #TODO: Find a better way to compare pyspark dfs
    if not processed_col.collect() == setup_dataframe.select(indiv_id_col).collect():
        errors_list.append("Individual identifier column {indiv_id_col} modified.")
    for degenerate_col in degenerate_cols:
        try: 
            processed_col = data_processor.process_single_column(degenerate_col).select(degenerate_col)
        except AnalysisException:
            processed_col = None
        if processed_col is not None:
            errors_list.append(
                f"Degenerate column {degenerate_col} not dropped from dataframe."
            )
    for categorical_col in categorical_cols:
        processed_col = data_processor.process_single_column(categorical_col)
        if not isinstance(processed_col.schema[categorical_col].dataType, StringType):
            errors_list.append(
                f"Categorical column {categorical_col} not cast to categorical dtype."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))    
    
def test_check_panel_consistency(setup_config, setup_dataframe):
    """Test that check_panel_consistency raises an error if an
    observation is duplicated.
    """
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    data_processor.data = data_processor.data.union(spark.createDataFrame(data_processor.data.take(1)))
    #TODO: Fix the iloc 1 line
    with pytest.raises(AssertionError):
        data_processor.check_panel_consistency()
        
def test_sort_panel_data(setup_config, setup_dataframe):
    """Test that sort_panel_data re-sorts scrambled observations."""
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    first_ten_rows_already_sorted = spark.createDataFrame(data_processor.data.take(10))
    data_processor.data = data_processor.data.orderBy(rand())
    data_processor.data = data_processor.sort_panel_data()
    first_ten_rows_scrambled_then_sorted = spark.createDataFrame(data_processor.data.take(10))
    assert first_ten_rows_scrambled_then_sorted.collect() == first_ten_rows_already_sorted.collect()

def test_flag_validation_individuals(setup_config, setup_dataframe):
    """Test that validation set is given share of observations and contains
    all observations of each individual therein.
    """
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    data_processor.data.select('_validation').show()
    error_tolerance = 0.1
    data_processor.data.printSchema()
    data_processor
    data_processor.data = data_processor.flag_validation_individuals()
    share_in_validation_sample = data_processor.data.withColumn('_validation', data_processor.data['_validation'].cast('int')).agg({'_validation':'sum'}).show()#.first()[0]
    share_approximately_correct = (
        (data_processor.config["VALIDATION_SHARE"] - error_tolerance)
        <= share_in_validation_sample
    ) and (
        share_in_validation_sample
        <= (data_processor.config["VALIDATION_SHARE"] + error_tolerance)
    )
    rates_individuals_within_validation_group = data_processor.data.groupBy(
        data_processor.config["INDIVIDUAL_IDENTIFIER"]
    )["validation"].mean()
    individual_consistently_in_validation_group = (
        rates_individuals_within_validation_group == 1
    ) | (rates_individuals_within_validation_group == 0)
    assert share_approximately_correct
    assert np.mean(individual_consistently_in_validation_group) == 1


SEED = 9999
np.random.seed(SEED)