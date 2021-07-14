

call C:/Users/%username%/Anaconda3/Scripts/activate.bat
call conda create -y -n fifeforspark python=3.7
call C:/Users/%username%/Anaconda3/Scripts/activate.bat fifeforspark
rmdir build /s /q
call conda install -y -c conda-forge shap iniconfig
call conda install -y -c anaconda tensorflow
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade setuptools wheel
pip install --trusted-host pypi.org hyperopt lifelines pandas pyspark seaborn jupyter jupyter_core ipykernel fife findspark --user
echo installed packages correctly!
pause