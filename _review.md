CO Review Comments

General feedback:
- Recommend making documentation more concise (e.g., you can provide links to sources that give background info
on Apache Spark, PySpark, and MMLSpark for users who may be unfamiliar, and allow users who are familiar to 
bypass)
- Recommend clarifying the installation/set-up process (e.g., if you recommend Databricks, provide detailed
instructions for how users can install MMLSpark and FIFEforSpark and attach to cluster in Databricks)

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
    a) `create_example_data1` function is very slow; also not clear to the user what the difference is between
	`create_example_data1` and `create_example_data2` functions

3) lgb_modelers.py
    a) Encountering the following error when running `LGBSurvivalModeler(data=data_processor.data).build_model()`:
	`java.lang.NoClassDefFoundError: org/apache/spark/ml/util/MLWritable$class`; note that importing 
	`fifeforspark.lgb_modelers` succeeds without triggering the warning that MMLSpark could not be imported;
	using 8.3 Databricks Runtime, mmlspark_2.11-1.0.0-rc3 Maven coordinates for MMLSpark; full traceback is in 
	[3a_error_buildmodel.txt](3a_error_buildmodel.txt)
    b) `tqdm` package is not included in the 8.3 Databricks Runtime, and trying to import LGBSurvivalModeler
	from fifeforspark.lgb_modelers returns following error: `ModuleNotFoundError: No module named tqdm`;
	not a huge issue as users can install it themselves from PyPI, but it could be helpful to list in one place
	all additional packages that FIFE users need to install to make the switch to FIFEforSpark