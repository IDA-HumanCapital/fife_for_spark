#### user_guide.md

- Is capitalization of config parameters, e.g. "learningRate" important?
- Do "other modelers" exist in F4S?
- I favor staying DRY by linking to FIFE documentation on non-Spark topics such as "Introduction to Survival Analysis," rather than keeping F4S "self-contained"
- The "life"/"death" discussion is more confusing than helpful; recommend focusing on the abstract notion of binary outcome
- Censoring is not just "one of the key concepts," but the *defining concept* of survival analysis"

- Censoring discussion has multiple incorrect statements
- Asterisks appearing around "Kaplan-Meier estimator"
- "This method is only as useful as the number of observations with each unique value *is large*"
- "we can forecast a future feature value" conflicts with earlier statement that FIFEforSpark focuses on survival analysis
- Clarify that gradient-boosted trees are not only for "binary classification and regression"
- Broken footnote in LightGBM section
- Errant quote marks in mathematical expression in LightGBM section
- Replace details on LightGBM and survival analysis with links
- Errant asterisk in "BY*FEATURE"

#### spark_help.md

- Change "model quality" to "model performance"
- Add vertical space between error message and following paragraph
- Delete Maven version suffix after "1.0.0-rc3"
- Fix or remove Maven repository link
- Replace link in footnote 5 with a link related to PyPI issue

#### fifeforspark (Modules)

- First three modules not showing classes and functions

#### \_\_main\_\_.py

- Change to match FIFE version, which executes forecast OR evaluate, not both
- Print each step's run time immediately after the step concludes, rather than all at the end
- Import functions and classes, not modules

#### processors.py

- Consider parallelizing the for loop in process_all_columns
- Consider sorting before processing to allowing skipping of orderBy operations

#### lgb_modelers.py

- Does transform_features expect self.data to be a Spark DataFrame? If so, should we use Spark dtype names?
- If save_model "Functionality is currently progress", the function should throw a NotImplementedError

#### gbt_modelers.py

- Refactor WET code out of lgb_modelers.train_single_model
- Incorporate lack of support for missing values into transform_features

#### base_modelers.py

- Replace "i.e." with "e.g."
- Does L447 select the matching predictions if L459 excluded one or more observations from actuals?
- L477 should accord with docstring
- StateModeler and ExitModeler should throw NotImplementedError
- Does capitalization matter in L527?
- Add `+ '_new'` to right keys In L536-7

