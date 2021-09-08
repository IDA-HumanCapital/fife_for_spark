rem | This batch file creates and tests a FIFE package
rem | You may wish to update the python and FIFE versions in this script
rem | This file requires an Anaconda (or Miniconda) installation
rem | You may need to specify your local path to activate.bat

@echo off
setlocal enabledelayedexpansion
call C:/Users/%username%/Miniconda3/Scripts/activate.bat
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat
call conda create -y -n fifeforspark python=3.7
call C:/Users/%username%/Miniconda3/Scripts/activate.bat fifeforspark
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat fifeforspark
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade setuptools wheel
rmdir build /s /q
python setup.py sdist bdist_wheel
call conda install -y -c conda-forge iniconfig
@rem pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade setuptools wheel
@rem pip install --trusted-host pypi.org hyperopt lifelines pandas pyspark seaborn jupyter jupyter_core ipykernel fife findspark pyarrow koalas databricks sphinx tqdm--user

for /F %%i in ('python setup.py --version') do set version=%%i
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade dist/fifeforspark-%version%-py3-none-any.whl black pytest

echo installed packages correctly!

black .

pause
