CO Review Comments

General feedback:

- Recommend making documentation more concise to make it easy for users to find the information they care about
(e.g., you can provide links to sources that give background info on Apache Spark, PySpark, and MMLSpark 
for users who may be unfamiliar, and allow users who are familiar to bypass; can also streamline the language
throughout)

- Recommend clarifying the installation/set-up process (e.g., if you recommend Databricks, provide detailed
instructions for how users can install MMLSpark and FIFEforSpark and attach to cluster in Databricks)

- Due to the obscurity of error messages in Spark, recommend creating a separate page in the documentation to compile
commonly encountered error messages, along with potential causes and fixes (doesn't have to be comprehensive, 
just a place for users to start troubleshooting; can add to the list as users encounter and resolve new errors)

- Is `rf_modelers.py` ready for primetime? I see it in the `random_forest` branch, but not on `main`. (Also, I suspect
that comment 4b (below) will apply to this module/class as well.)


1) README.md

    a) Not obvious who "their" refers to in this statement

    b) It would be very helpful for users if you provide instructions for how to install MMLSpark and how 
	to attach it to a cluster in Databricks (or at least link to the SynapseML repository on Github)
	(edit: found some instructions in the formal docs which were helpful, but think it would still be 
	good to include in the README)

    c) Recommend including instructions for how to install FIFEforSpark in the README

    d) This statement is confusing to me; is it best to use FIFEforSpark in Databricks, or in a Python IDE?
	In either case, it's not clear how to get set up and install FIFEforSpark and all of the required dependencies

    e) I get a "ModuleNotFoundError: No module named 'findspark'" error message when running `import findspark`
	in Databricks (using Databricks Runtime 8.3/Spark 3.1.1/Scala 2.12)

    f) This code block is mostly redundant (and user has to spend time trying to identify the difference);
	recommend eliminating all but the line that applies to Databricks
	(e.g., `data_processor = PanelDataProcessor(data = spark.sql("select * from your_table"))`)

2) utils.py

    a) `create_example_data1` function is slow; also not clear to the user what the difference is between
	`create_example_data1` and `create_example_data2` functions

3) lgb_modelers.py

    a) [9/10 Update] Per Ed's recommendations, switching to 9.0 Databricks Runtime and 
	com.microsoft.ml.spark:mmlspark_2.12:1.0.0-rc3-59-bf337941-SNAPSHOT Maven coordinates resolved the error.
	These coordinates are not the most recent listed on the [MMLSpark GitHub repository](https://github.com/microsoft/SynapseML)
	though--recommend addressing in the documentation (in a general way, e.g. "encountering the following error may
	indicate an incompatibility issue between the Databricks runtime and maven repository", rather than
	trying to list all combinations of runtimes and coordinates that do/don't work)

	Encountering the following error when running `LGBSurvivalModeler(data=data_processor.data).build_model()`:
	`java.lang.NoClassDefFoundError: org/apache/spark/ml/util/MLWritable$class`; note that importing 
	`fifeforspark.lgb_modelers` succeeds without triggering the warning that MMLSpark could not be imported;
	using 8.3 Databricks Runtime, mmlspark_2.11-1.0.0-rc3 Maven coordinates for MMLSpark; full traceback is in 
	[3a_error_buildmodel.txt](3a_error_buildmodel.txt)

    b) `tqdm` package is not included in the 8.3 Databricks Runtime, and trying to import LGBSurvivalModeler
	from fifeforspark.lgb_modelers returns following error: `ModuleNotFoundError: No module named tqdm`;
	not a huge issue as users can install it themselves from PyPI, but it could be helpful to list in one place
	all additional packages that FIFE users need to install to make the switch to FIFEforSpark

4) gbt_modelers.py

    a) Update docstring of `train_single_model` function in `GBTModeler` class to remove reference to LightGBM

    b) Assuming that GBTModeler was implemented to provide an alternative to LightGBM (due to the difficulties of
	installing MMLSpark), it's a bit strange that GBTModeler inherits from the LGBModeler class, and that 
	you therefore need to import the LGBModeler class from fifeforspark.lgb_modelers for this module to work. 
	Is there an easy way to uncouple the LGBModeler and GBTModeler class? Alternatively, do they really need 
	to be two separate classes, or should there be one "tree-based modeler" class to which users can pass
	an argument for whether they want to use LightGBM or pyspark GBTClassifier (under the hood)?

5) base_modelers.py

    a) Is the attribute set in line 176 (`self.spark = SparkSession.builder.getOrCreate()`) used anywhere else?

6) fife-for-spark.readthedocs.io

    a) Recommend removing instructions for installing fifeforspark with firewall 
	[here](https://fife-for-spark.readthedocs.io/en/latest/user_guide.html) (ability to set trusted host 
	may depend on user's particular circumstances, e.g., security considerations)

    b) Recommend moving "Introduction to Survival Analysis" to separate page, and linking to it from the 
	Quickstart section. Would also recommend the same for "Configuration Parameters" 
	(with the goal of organizing information in such a way that users with varying levels 
	of familiarity can quickly navigate through the information and find what is most relevant to them)

    c) Recommend removing footnotes from documentation and linking directly to the sources in the text 
	instead (i.e., in-line)

    d) The link to the image of the MMLSpark_Maven coordinates 
	([here](https://fife-for-spark.readthedocs.io/en/latest/spark_help.html#how-to-download-mmlspark))
	directs me to a webpage with the message "SORRY This page does not exist yet."

    d) Nitpick: user guide notes that the model produced by FIFEforSpark is a list of models, exactly like FIFE
	([here](https://fife-for-spark.readthedocs.io/en/latest/user_guide.html#lgbsurvivalmodeler))--this is 
	not necessarily the case if FIFE users are using non-tree-based modelers