"""Conduct unit testing for fife.processors module."""
import sys, os
sys.path.append(os.path.abspath("C:/Users/jwang/Documents/GitHub/fife_for_spark/fifeforspark"))
import processors
import numpy as np
import pandas as pd
import pytest
import findspark
from pyspark.sql import SparkSession
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
    processed_col = data_processor.process_single_column(indiv_id_col)
    if not processed_col == setup_dataframe[indiv_id_col]:
        errors_list.append("Individual identifier column {indiv_id_col} modified.")
    for degenerate_col in degenerate_cols:
        processed_col = data_processor.process_single_column(degenerate_col)
        if processed_col is not None:
            errors_list.append(
                f"Degenerate column {degenerate_col} not dropped from dataframe."
            )
    for categorical_col in categorical_cols:
        processed_col = data_processor.process_single_column(categorical_col)
        if not isinstance(processed_col.dtypes, pd.api.types.CategoricalDtype):
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
    #Fix the iloc 1 line
    with pytest.raises(AssertionError):
        data_processor.check_panel_consistency()
        


SEED = 9999
np.random.seed(SEED)