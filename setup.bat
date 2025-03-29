@echo off
setlocal

:: Check if 'opencda' conda environment exists. If not, create it from environment.yml.
:: Usage: Double-click or run in CMD.
:create_env
:: Search for 'opencda' in conda env list
for /f "tokens=1" %%i in ('conda env list ^| find "opencda"') do (
    if "%%i" == "opencda" (
        echo Conda environment 'opencda' already exists. Skipping creation...
        goto :activate_env
    )
)

:: If 'opencda' not found, create it
echo Conda environment 'opencda' not found. Creating now...
call conda env create -f environment.yml

:: Check if conda env creation succeeded
if %errorlevel% neq 0 (
    echo ❌ Failed to create conda environment! Check environment.yml.
    pause
    exit /b 1
)

:: Activate the environment
:activate_env
call conda activate opencda
if %errorlevel% neq 0 (
    echo Failed to activate conda environment!
    pause
    exit /b 1
)

echo Environment setup complete! Current env: opencda. Now Insatll the requirements...
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --no-deps
pip install -r ./requirements.txt

echo Requirements installed! Now Set up OpenCDA...
python setup.py develop

:setup_carla
:: Check Python version (Windows typically uses just Python, not python3)
python --version
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)
echo Checking if carla-%CARLA_VERSION% is installed...
python -c "import carla" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ carla-%CARLA_VERSION% is already installed. Skipping installation steps.
    goto setup_opencood
)

:: Check if CARLA_HOME is set
if "%CARLA_HOME%"=="" (
    echo Error: Please set CARLA_HOME before running this script
    exit /b 1
)

:: Set default CARLA_VERSION if not specified
if "%CARLA_VERSION%"=="" (
    set CARLA_VERSION=0.9.11
)


:: Path to the CARLA egg file
set CARLA_EGG_FILE="%CARLA_HOME%\PythonAPI\carla\dist\carla-%CARLA_VERSION%-py3.7-win-amd64.egg"
if not exist %CARLA_EGG_FILE% (
    echo Error: %CARLA_EGG_FILE% cannot be found. Please make sure you are using python3.7 and carla %CARLA_VERSION%
    exit /b 1
)

:: Set cache directory
set CACHE=%CD%\cache
if not exist "%CACHE%" (
    echo creating cache folder for carla PythonAPI egg file
    mkdir "%CACHE%"
)

echo copying egg file to cache folder
copy %CARLA_EGG_FILE% "%CACHE%"

echo unzip egg file
:: On Windows, 'tar' can be used for extraction (Windows 10+)
if exist "%CACHE%\carla-%CARLA_VERSION%-py3.7-win-amd64.egg" (
    tar -xf "%CACHE%\carla-%CARLA_VERSION%-py3.7-win-amd64.egg" -C "%CACHE%"
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to extract egg file. Make sure you have tar or another extraction tool available.
        exit /b 1
    )
)

:: Rename the extracted folder to match Linux version pattern for compatibility
if exist "%CACHE%\EGG-INFO" (
    move "%CACHE%\EGG-INFO" "%CACHE%\carla-%CARLA_VERSION%-py3.7-win-amd64"
)

echo copy setup file to egg folder
set SETUP_PY=%CD%\scripts\setup.py
if exist "%SETUP_PY%" (
    copy "%SETUP_PY%" "%CACHE%"
) else (
    echo Warning: setup.py not found at %SETUP_PY%
)

echo Successful! Now install carla into your python package
:: Activate conda environment (adjust if your environment name is different)
call conda activate opencda
if %ERRORLEVEL% neq 0 (
    echo Warning: Failed to activate conda environment 'opencda'
)
pip install -e "%CACHE%"

:setup_opencood
echo Successful! Now install opencood into your python package
:: Activate conda environment (adjust if your environment name is different)
call conda activate opencda
if %ERRORLEVEL% neq 0 (
    echo Warning: Failed to activate conda environment 'opencda'
)
cd ./opencood
pip install -r ./requirements.txt
python  ./setup.py develop
python ./opencood/utils/setup.py build_ext --inplace

pause
endlocal
