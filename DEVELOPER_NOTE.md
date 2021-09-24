

## Developer Note:

Welcome to the developer note for FIFEforSpark. The purpose of this document is to detail the current status of development and describe some of the changes we wish to implement in the future. Additionally, you can find the link to the JIRA page [here](https://sfrdtps.ida.org/jira/projects/FFS/issues/FFS-41?filter=allopenissues), where we detail some of the issues/bugs in greater detail.

#### Current Status of Development:

As of now, FIFEforSpark offers two different modeling options: Gradient Boosted Trees (LightGBM and PySpark ML) and Random Forests (Pyspark ML). Using these models, as well as the distributed version of the panel data processor, FIFEforSpark is able to replicate the core survival analysis functionality of FIFE. Many of the functions we wrote are distributed or "Sparkified" in nature; however, some are intentionally left as processes on the driver node as they do not necessitate the cluster and can run significantly faster when performed on a single node. We have implemented unit tests, have two example notebooks and have thorough documentation on all of the modules and related subject matter. 

#### Ongoing issues:

That being said, there are several notable differences that we have investigated but could not implement. 

First, we could not manage to get a non-grid search hyper-parameter optimization routine into FIFEforSpark. Packages like Optuna and Hyperopt appear to have the functionality, but due to a lack of documentation and information, we could not identify whether they worked with the models we implement (our tests did not produce fruitful results either). We decided against a grid search of any kind as just a single iteration of the model takes a very long time.

Second, we attempted to look at early stopping. Based on our preliminary analysis, early stopping made the model building process take a longer time. Due to this not matching our hypotheses, we are holding off on merging that branch into the master branch until we can complete more tests.

Third, the GBT and RF modelers (which use PySpark ML) work correctly in Jupyter Notebook with respect to missing values; however, within Databricks, both throw an error on the REIGN dataset when PCNTILE_RISK is included. We are not sure why that is. Removing that feature, or subsetting the dataset to non-missing values results in a completed model run. 

#### Future Areas of Development:

There are several things we wish to implement in the future but are not yet able to do.

- Testing in a proper cluster.
  - We have only tested FIFEforSpark in Databricks' Community Edition and the PySpark local machine implementation. As such, FIFEforSpark has not been used to run anything that FIFE itself cannot run. We hope to be able to test this on a cluster that can sustain a much larger dataset.
- Adding more documentation on common Spark errors and installation instructions
  - A few reviewers have asked us to include references or documentation pages with greater detail on common Spark errors as well as more thorough installation instructions for dependencies and for FIFEforSpark. 
- Improving function efficiency
  - One noticeable change between FIFE and FIFEforSpark from the user side is the amount of time the user must wait after calling a function. While FIFE is almost immediate for several of these function calls, FIFEforSpark is not. We are constantly searching for ways to improve the overall efficiency of different functions and algorithms
- Build a better command line interface and include the relevant documentation
  - A large part of FIFE is the ability to run it from the command line. While this exists in FIFEforSpark, we seek to improve the functionality
- Add visualizations
  - FIFE produces many plots automatically that FIFEforSpark may eventually be able to implement. This has not yet been attempted, though it remains an item of interest.
- Adding in simultaneous runs of an RPM
  - One idea we had during the initial phases of the project was to allow FIFEforSpark to run FIFE in parallel on multiple different nodes. While the data used by FIFE could be housed on the driver node, having each executor run its own single-node FIFE model could allow for a user to test many variations of a model in a shorter amount of time.
- Adding in other modelers
  - While FIFE supports the State and Exit modeler, FIFEforSpark currently does not. Hopefully that is a change that we can make in the near future
- Restructuring the GBT/RF/LGB classes. Currently, gbt_modelers and rf_modelers both inherit lgb_modelers as a way to reduce unnecessary lines of code and modules. We seek to replace this with either a main tree_modeler abstract parent module with 3 descendants, or a single module with a parameter for the type of model.

Sincerely,

Akshay and Ed