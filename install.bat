@echo off

:: Install the main project in editable mode 
pip install -e .
 
:: Navigate to the 'llamafactory' directory and install it in editable mode  
cd nanograd\trainer
pip install --no-deps -e .

:: Navigate back to the original directory
cd ..
