## Quick Start

The Finite-Interval Forecasting Engine for Spark (FIFEforSpark) is an adaptation of the Finite-Interval Forecasting Engine for the Apache Spark environment. Currently, it provides machine learning models, specifically a gradient boosted tree model, for discrete-time survival analysis. We will show you here how to build your own model pipeline in Python. Soon, we will be able to also use the pre-created functionality in the command line. You can also see a full example that uses the [Rulers, Elections, and Irregular Governance dataset (REIGN)](https://oefdatascience.github.io/REIGN.github.io/) in this notebook.

### Installation

``` python
pip install fifeforspark
```

Details

- Install an [Anaconda distribution](https://www.anaconda.com/distribution/) of Python version 3.8.5 or later
- From pip (with no firewall):
  - Open Anaconda Prompt
  - Execute `pip install fifeforspark`
- From pip (with firewall):
  - Open Anaconda Prompt
  - Execute `pip install --trusted-host pypi.org fifeforspark`

Alternatives

- From pypi.org ():

  - Download the `.whl` or `.tar.gz` file

  - Open Anaconda Prompt

  - - Change the current directory in Anaconda Prompt to the location where the `.whl` or `.tar.gz` file is saved. 

      Example: `cd C:\Users\insert-user-name\Downloads`

  - - Pip install the name of the `.whl` or `.tar.gz` file. 

      Example: `pip install fife-1.5.1-py3-none-any.whl`

- From GitHub (https://github.com/IDA-HumanCapital/fife_for_spark):

  - Clone the fife_for_spark repository
  - Open Anaconda Prompt
  - Change the current directory in Anaconda Prompt to the directory of the cloned FIFE repository
  - Execute `python setup.py sdist bdist_wheel`
  - Execute `pip install dist/fife_for_spark-0.0.1-py3-none-any.whl`

### LGBSurvivalModeler

To get started, we'll want a module to process our panel data and another module to use the processed data to train a model. In this example, we'll build gradient-boosted tree models using [MMLSpark's LightGBM](https://github.com/microsoft/SynapseML/blob/master/docs/lightgbm.md). 

```python
from fifeforspark.processors import PanelDataProcessor
from fifeforspark.lgb_modelers import LGBSurvivalModeler
```

We’ll need to supply a data file in the form of panel data. If you don’t have a file already available, you can use our example data. This is nearly identical to FIFE's create_example_data function, though there are two versions available.

```python
from fifeforspark.utils import create_example_data1
data = create_example_data1()
```

In order to make sure our results are reproducible, we can use the `make_results_reproducible` function: 

```python
from fifeforspark.utils import make_results_reproducible
make_results_reproducible()
```

Exactly like FIFE, the Panel Data Processor will make our dataset ready for modeling, to include sorting by individual and time, dropping degenerate and duplicated features, and computing survival durations and censorship status. We can specify processing parameters different than the defaults in the `config` argument.    

```python
data_processor = PanelDataProcessor(data=data)
data_processor.build_processed_data()
```

Once we've called `data_processor.build_processed_data()`, the completed data will be available at `data_processor.data`. We can then use this in our models. For our first example, let's use a gradient-boosted trees modeler. Like the data building process, we can change modeling parameters in the config argument. Perhaps we prefer a learning rate value of 0.2 rather than the default value of 0.1.

```python
survival_modeler = LGBSurvivalModeler(config={'LEARNING_RATE': 0.2}, 
    data=data_processor.data)
survival_modeler.build_model()
```

Exactly like FIFE, our discrete-time survival "model" is actually a list of models, one for each time horizon. The first model produces a probability of survival through the first future period, the second model produces a probability of survival through the second future period conditional on survival through the first future period, and so on. In general, we are most interested in forecasts for individuals that are still in the population in the final period of the data. The `forecast` method gives us exactly those forecasts. The survival probabilities in the forecasts are cumulative - they are not conditional on survival through any prior future period.     

```python
survival_modeler.forecast()
```

Because our forecasts are for the final period of data, we can't measure their performance in future periods. However, we can train a new model where we pretend an earlier period is the final period, then measure performance through the actual final period. In other words, we can reserve some of the most recent periods for testing. Notice once again that the syntax is identical to that of FIFE. Here is an example:

```python
test_intervals = 4
data_processor = PanelDataProcessor(config={'TEST_INTERVALS': test_intervals},
                                	data=data)
data_processor.build_processed_data()
survival_modeler = LGBSurvivalModeler(data=data_processor.data)
survival_modeler.build_model()
```

The `evaluate` method offers a suite of performance metrics specific to each time horizon as well as the concordance index over the restricted mean survival time. We can pass a Boolean mask to `evaluate` to obtain metrics only for the period we pretended was the most recent period. Be careful to not set `TEST_INTERVALS` to be too large; if it exceeds half of the number of intervals, it will not provide the training dataset with enough intervals. Here, rather than passing in a Boolean pandas series, we pass in a Spark DataFrame column with Boolean values.
    

```python
min_val = survival_modeler.data.select(survival_modeler.data['_period']).agg({'_period': 'min'}).first()[0]
evaluation_subset = survival_modeler.data.select(survival_modeler.data['_period'] == min_val)   
survival_modeler.evaluate(evaluation_subset)
```

`evaluate` offers just one of many ways to examine a model. For example, we can answer "In each time period, what share of the observations two periods past would we expect to still be around?"

```python
survival_modeler.tabulate_retention_rates(2)
```

Other modelers define different ways of using data to create forecasts and metrics, but they all support the methods `build_model`, `forecast`, `evaluate` and more.

## Introduction to Survival Analysis

### Panel Data

The following is also all true for FIFE; however, to keep this package self contained, we include it here as well.

FIFEforSpark uses data called **panel data**, which we define as periodic observations of subjects within a given time frame. Each observation is defined by a subject and a period. For example, each record of an active duty service member has a unique combination of social security number and year. Each observation has a vector of feature values. We may observe pay grade, occupation, and duty location. The period of observation is also a feature value. Not every service member is observed in every year, which makes the panel **unbalanced**. 

Unbalanced panels are not unique to personnel data, but are a natural product of any periodic record-keeping where the roster of subjects  changes over periods. An unbalanced panel could track equipment, medical patients, organizations, research and development programs, stockpiles, or natural phenomena. 

Suppose you have a dataset that looks like this:

| ID   | period | feature_1 | feature_2 | feature_3 | ...  |
| ---- | ------ | --------- | --------- | --------- | ---- |
| 0    | 2016   | 7.2       | A         | 2AX       | ...  |
| 0    | 2017   | 6.4       | A         | 2AX       | ...  |
| 0    | 2018   | 6.6       | A         | 1FX       | ...  |
| 0    | 2019   | 7.1       | A         | 1FX       | ...  |
| 1    | 2016   | 5.3       | B         | 1RM       | ...  |
| 1    | 2017   | 5.4       | B         | 1RM       | ...  |
| 2    | 2017   | 6.7       | A         | 1FX       | ...  |
| 2    | 2018   | 6.9       | A         | 1RM       | ...  |
| 2    | 2019   | 6.9       | A         | 1FX       | ...  |
| 3    | 2017   | 4.3       | B         | 2AX       | ...  |
| 3    | 2018   | 4.1       | B         | 2AX       | ...  |
| 4    | 2019   | 7.4       | B         | 1RM       | ...  |
| ...  | ...    | ...       | ...       | ...       | ...  |

The entities with IDs 0, 2, and 4 are observed in the dataset in 2019.

While FIFE offers a significantly larger suite of models designed to answer a variety of questions, FIFEforSpark is mainly focused on one question: what are each of their probabilities of being observed in any future year? Fortunately, FIFEforSpark can estimate answers to these questions for any unbalanced panel dataset.

Survival analysis was first used to determine if someone lived or died. Today, however, we can use its theories to define exits from a binary state as either a _life_ or _death_. For instance, we can investigate if an individual leaves the military service by asking if this individual "dies." If another individual joins the service, we can say they were "born."

We can use these ideas to define retention as "living" and attrition as "dying." To investigate retention, FIFEforSpark seeks forecasts for observations in the final period of the dataset. For each observation, we define “retention” as the observation of the  same subject in a given number of consecutive future periods. For example, 2-year retention of a subject observed in 2019 means observing the same subject in 2020 and 2021. Note that we define  retention with reference to the time of observation. We ask not “will this individual be in the dataset in 2 years,” but “will this member be in this dataset 2 more years from this date?”

### Censoring

One of the defining concepts of survival analysis is **censoring**. If an individual is present in the last period for which we have data, the individual is *right censored*, as we can't identify when that individual left the dataset. Conversely, an individual is *left censored* if they are present in the first period for which we have data, though that is less of a concern for FIFEforSpark.

### Survival Function

The survival function is the true, unknown, curve that describes the amount of the population remaining at time :math:`t`. This function, :math:`S(t)`, is defined as:

.. math::

    S(t)=Pr(T>t)

Thus, this function notes the probability that an individual does not leave at time :math:`t`, or the probability of staying past time :math:`t`. This is a decreasing function. 

### Hazard Function

Let :math:`y(t)` be 1 if a given individual exits an amount of time :math:`t` into the future, and 0 otherwise. Let :math:`h(t)` be the probability that an individual exits at time :math:`t`, given the individual has not exited before. :math:`h(t)` is a **hazard function**, which takes :math:`t` as input and outputs a probability that :math:`y(t)` equals 1. It can also be shown that: 

.. math:: 

    h(t) = \frac{-S'(t)}{S(t)}

Data that allows us to observe :math:`y` for different values of :math:`t` inform us about which hazard functions will better match true outcomes (thus, have better _performance_). We _fit_ a model by using data to estimate the function that matches the true data the best.

### Traditional Methods for Fitting a Survival Curve

The hardest part about fitting a hazard function is accounting for censoring. A particularly famous and simple method of fitting a forecasting model  under censoring is the [Kaplan-Meier estimator](https://www.tandfonline.com/doi/abs/10.1080/01621459.1958.10501452). For each time horizon at  which at least one observation exited, the Kaplan-Meier estimator  computes the share of observations that exited among those observed to  exit or survive at the given time horizon. The Kaplan-Meier estimator is defined as: 

.. math::

    \hat{S}(t) = \prod_{t_{i}<t}\frac{n_i-d_i}{n_i}

where :math:`d_i` are the number of attritions at time :math:`t` and :math:`n_i` is the number of individuals who are likely to attrition just before :math:`t`. This survival curve estimator describes share retained ("alive") as a function of the time horizon. The Kaplan-Meier estimator is the most famous estimation to the true survival curve in traditional survival analysis.

#### Estimation using Feature Values

The problem is the Kaplan-Meier estimation to the survival curve, and the resulting hazard function, do not account for the vector of feature values, :math:`x`. Without accounting for :math:`x`, the hazard function is the same for all observations, so we cannot  estimate which subjects will retain longer, and we cannot target retention interventions. There are other ways to estimate hazards and survival curves using feature values. Parametric estimators account for :math:`x` by specifying a functional form for :math:`h`. For example, **proportional hazards regression** specifies that :math:`h` is the product of a baseline hazard :math:`h_{0}(t)` and :math:`e` raised to the inner product of :math:`x` and some constant vector :math:`\beta`. The principal drawback of specifying a functional form is that it limits how well a model can fit to data.

Another method to account for :math:`x` is to apply the Kaplan-Meier estimator separately for each unique value of :math:`x`. We can describe this method as :math:`h(t,x)=h_{x}(t)`. This method is only as useful as the number of observations with each unique value. For example, because there are thousands of active duty service members in each service in each year, this method would be  useful to estimate a separate hazard function for each service. We could also estimate a separate hazard function for each combination of service and entry year. However, we would be limiting :math:`x` to contain the values of only two features. We could not use this method to estimate a separate hazard function for each combination of  feature values among the hundreds of features in our data. Even a single feature with continuous support, such as the amount a member commits to their Thrift Savings Plan (TSP), makes this method infeasible.

### FIFEforSpark : Using Machine Learning

With FIFEforSpark, we can use machine learning to consider the interactions of feature values in predicting whether an individual will stay or leave. Traditional methods in the domain of survival analysis effectively consider all forecastable future time horizons by expressing the probability of retention as a continuous function of time since observation. In the case of panel data,  however, all forecastable time horizons fall in a discrete domain with a finite number of intervals. We can, therefore, consider each forecastable time horizon separately rather than treating time as continuous, unlocking all manner of forecasting methods not designed for survival analysis, including state-of-the-art machine learning methods.

Our method to account for :math:`x` is to estimate the hazard separately for each unique value of :math:`t`. We can describe this method as :math:`h(t,x)=h_{t}(x)`. This method is only as useful as the number of observations that exit at each time horizon. If the time horizon has continuous support, so that there are an infinite number of forecastable time horizons, this method is infeasible. However, panel data exhibit a finite number of forecastable time horizons. A panel data time horizon must be a whole  number of periods less than the number of periods in the dataset.  Therefore we can fit :math:`h_{t}(x)` separately for each such :math:`t`.

For each :math:`t`, we fit :math:`h_{t}(x)` to observations of the feature vector :math:`x` and the binary outcome :math:`y`. We can use any binary classification algorithm to fit :math:`h_{t}(x)`. We need not specify a functional form. Using FIFEforSpark, we can fit not only the hazard, but any function of :math:`x` for each time horizon :math:`t`. For example, we can forecast the future value of some feature  immediately before a given time horizon, conditional on exit at that  time horizon. Alternatively, we can forecast a future feature value conditional on survival. 

Like the Kaplan-Meier estimator, our method addresses censoring by  considering only observations for which we observe exit or survival at the given time horizon. In other words, we address censoring by subsetting the data for each :math:`t`. The panel nature of the data make this subsetting simple: if there are :math:`T` periods in the data, only observations from the earliest :math:`T−t` periods allow us to observe exit or survival :math:`t` periods ahead. For example, suppose we have annual observations from 2000 through 2019 available for model fitting. Let our time horizon be four years. We exclude the most recent four years, leaving observations from 2000 through 2015. Among those observations, we keep only those retained at least three years from the time of observation. Then we compute the binary outcome of survival or exit four years from the time of observation. This estimation actually looks at one minus the hazard, which leaves survival as a positive class. 

#### Gradient-Boosted Trees

Gradient-boosted trees are a state-of-the-art machine learning algorithm for binary classification and regression, along with several other use cases. A gradient-boosted trees model is a sequence of **decision trees**. Each decision tree repeatedly sorts observations based on a binary condition. For example, a decision tree for predicting one-year Army officer retention could first sort observations into those with fewer than six  years of service and those with six or more. Then it could sort the former subgroup into those who accessed through the Unites States Military Academy and those who accessed otherwise. The tree could sort the subgroup with six or more years of service into those who are in the Infantry Branch or Field Artillery Branch and those who are not. The tree could continue to sort each subgroup until reaching some stopping criterion. Any given observation sorts into one of the resulting subgroups (**leaves**), and is assigned the predicted value associated with that leaf.

.. image:: images/ex_dec_tree.png
    :width: 600px
    :align: center

*An example decision tree for predicting attrition of a first year Army Officer.*

In the model training process, we use training data to decide what  conditions to sort by and what values to predict for each leaf. We provide a cursory description of the model training process here to familiarize the reader and explain our modeling choices. Our description omits many caveats, intricacies, and techniques developed by the  machine learning community over years of research. We refer the advanced reader to the documentation for [LightGBM](https://lightgbm.readthedocs.io/en/latest/) for details on the particular software we use to train gradient-boosted tree models.

We begin model training with the entire training data. We sort the data into two subgroups using the feature and feature value that most informatively divides the training data into those who were retained and those who were not. We apply the same information criterion to split each of the resulting subgroups. We stop splitting after reaching any of our stopping criteria, such as a maximum number of splits.

Each leaf of a trained tree contains one or more training observations; the predicted value associated with a leaf is the mean outcome among those training observations. In our example, the mean outcome is the share of Army officers retained. Look at the leaf of the tree which represents Officers with less than six years of service not from the Military Academy and from a state that is not Virginia.<sup>1</sup> If 90% of such Officers in the training data were retained, the tree will assign a retention probability of 90% to any Officer that falls into the same leaf. If we were just looking for binary classification, this would be assigned to "remain."

While we could stop training after one tree, gradient boosting allows us to improve model performance by training more trees to correct model errors. A prediction from a gradient-boosted tree model is a weighted sum of predicted values from the trained trees. Since our binary classification call,  **LGBSurvivalModeler**, uses boosted tree models, the cumulative product of the predictions form an estimated survival function. The survival probabilities at time horizon :math:`t` periods into the future are defined as: 

.. math::

    Pr(T_{i\tau}\ge t|X_{i\tau}) 

where :math:`X_{i\tau}` is a vector of feature values for individual :math:`i` at time :math:`\tau`, and :math:`T_{i\tau}` is the number of consecutive future periods the individual remains after time :math:`\tau`. 

## Configuration Parameters

FIFEforSpark is highly customizable for an individual project. These are the different parameters that can be overwritten in a `config.json` file.

### Input/output

DATA_FILE_PATH; default: `"Input_Data.csv"`; type: String
	A relative or absolute path to the input data file.
NOTES_FOR_LOG; default: `"No config notes specified"`; type: String
	Custom text that will be printed in the log produced during execution.
RESULTS_PATH; default: `"FIFE_results"`; type: String
	The relative or absolute path on which Intermediate and Output folders will be stored. The path will be created if it does not exist.

### Reproducibility

SEED; default: 9999; type: Integer
	The initializing value for all random number generators. Strongly recommend not changing; changes will not produce generalizable changes in performance.

### Identifiers

INDIVIDUAL_IDENTIFIER; default: `""` (empty string); type: String
	The name of the feature that identifies individuals that persist over multiple time periods in the data. If an empty string, defaults to the leftmost column in the data.
TIME_IDENTIFIER; default: `""` (empty string); type: String
	The name of the feature that identifies time periods in the data. If an empty string, defaults to the second-leftmost column in the data.

### Feature types

CATEGORICAL_SUFFIXES; default: `[]` (empty list); type: List of strings
	Optional list of suffixes denoting that columns ending with such a suffix should be treated as categorical. Useful for flagging categorical columns that have a numeric data type and more than MAX_NUM_CAT unique values. Column names with a categorical suffix and a numeric suffix will be identified as categorical.
DATETIME_AS_DATE; default: `true`; type: Boolean
	How datetime features will be represented for the gradient-boosted trees modeler. If True, datetime features will be converted to integers in YYYYMMDD format for gradient-boosted trees modeler. Otherwise, datetime features will be converted to nanoseconds.
MAX_NULL_SHARE; default: 0.999; type: Decimal
	The maximum share of observations that may have a null value for a feature to be kept for training. Larger values may increase run time, risk of memory error, and/or model performance
MAX_UNIQUE_NUMERIC_CATS; default: 1024; type: Integer
	The maximum number of unique values for a feature of a numeric type to be considered categorical. Larger values may increase or decrease performance and/or increase run time.
NUMERIC_SUFFIXES; default: `[]` (empty list); type: List of strings
	Optional list of suffixes denoting that columns ending with such a suffix should be treated as numeric. Useful for flagging columns that have a numeric data type and fewer than MAX_NUM_CAT unique values. Column names with a categorical suffix and a numeric suffix will be identified as categorical.

### Training set

MIN_SURVIVORS_IN_TRAIN; default: 64; type: Integer
	The minimum number of training set observations surviving a given time horizon for the model to be trained to make predictions for that time horizon.
TEST_INTERVALS; default: -1; type: Integer
	The number of most recent periods to treat as absent from the data during training for the purpose of model evaluation. Larger values may decrease model performance and run time and/or increase evaluation time frame.
TEST_PERIODS; default: 0; type: Integer
	One plus the value represented by TEST_INTERVALS. Deprecated and overridden by TEST_INTERVALS.
VALIDATION_SHARE; default: 0.25; type: Decimal
	The share of observations used for evaluation instead of training for hyperoptimization or early stopping. Larger values may increase or decrease model performance and/or run time.

### Modeler types

TREE_MODELS; default: `true`; type: Boolean
	Whether FIFE will train gradient-boosted trees, as opposed to a neural network.


### Metrics

BY\_FEATURE; default: default: `""` (empty string); type: String
	The name of the feature for which FIFE will also produce separate Metrics_{value}.csv files for each group defined by each unique value of that feature.
QUANTILES; default: 5; type: Integer
	The number of similarly-sized bins for which survival and total counts will be reported for each time horizon. Larger values may increase run time and/or evaluation precision.
RETENTION_INTERVAL; default: 1; type: Integer
	The number of periods over which retention rates are computed.

## Frequently Asked Questions
Note: FAQs are coming soon.