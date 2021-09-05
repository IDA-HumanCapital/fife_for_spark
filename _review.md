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


