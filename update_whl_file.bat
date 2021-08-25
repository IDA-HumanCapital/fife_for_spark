@echo off

call C:/Users/%username%/Miniconda3/Scripts/activate.bat
call C:/Users/%username%/AppData/Local/Continuum/anaconda3/Scripts/activate.bat

python setup.py sdist bdist_wheel

pause
