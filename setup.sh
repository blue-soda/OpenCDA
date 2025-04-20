#!/bin/bash

# Check if 'opencda' conda environment exists. If not, create it from environment.yml.
echo "Checking if 'opencda' conda environment exists..."
if conda env list | grep -q "opencda"; then
    echo "Conda environment 'opencda' already exists. Skipping creation..."
else
    echo "Conda environment 'opencda' not found. Creating now..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create conda environment! Check environment.yml."
        exit 1
    fi
fi

# Activate the environment
echo "Activating 'opencda' environment..."
# source $(conda info --base)/etc/profile.d/conda.sh
conda activate opencda
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment!"
    exit 1
fi

echo "Environment setup complete! Current env: opencda. Now installing requirements..."
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --no-deps
pip install -r ./requirements.txt

echo "Requirements installed! Now setting up OpenCDA..."
python setup.py develop

# Check Python version
echo "Checking Python version..."
python --version
if [ $? -ne 0 ]; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if CARLA is installed
echo "Checking if carla-$CARLA_VERSION is installed..."
python -c "import carla" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ carla-$CARLA_VERSION is already installed. Skipping installation steps."
else
    # Check if CARLA_HOME is set
    if [ -z "$CARLA_HOME" ]; then
        echo "Error: Please set CARLA_HOME before running this script"
        exit 1
    fi

    # Set default CARLA_VERSION if not specified
    if [ -z "$CARLA_VERSION" ]; then
        CARLA_VERSION="0.9.11"
    fi

    # Path to the CARLA egg file
    CARLA_EGG_FILE="$CARLA_HOME/PythonAPI/carla/dist/carla-$CARLA_VERSION-py3.7-linux-x86_64.egg"
    if [ ! -f "$CARLA_EGG_FILE" ]; then
        echo "Error: $CARLA_EGG_FILE cannot be found. Please make sure you are using python3.7 and carla $CARLA_VERSION"
        exit 1
    fi

    # Set cache directory
    CACHE="$(pwd)/cache"
    if [ ! -d "$CACHE" ]; then
        echo "Creating cache folder for carla PythonAPI egg file"
        mkdir -p "$CACHE"
    fi

    echo "Copying egg file to cache folder"
    cp "$CARLA_EGG_FILE" "$CACHE"

    echo "Unzipping egg file"
    unzip -q "$CACHE/carla-$CARLA_VERSION-py3.7-linux-x86_64.egg" -d "$CACHE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract egg file. Make sure you have unzip installed."
        exit 1
    fi

    # Rename the extracted folder to match Linux version pattern for compatibility
    if [ -d "$CACHE/EGG-INFO" ]; then
        mv "$CACHE/EGG-INFO" "$CACHE/carla-$CARLA_VERSION-py3.7-linux-x86_64"
    fi

    echo "Copying setup file to egg folder"
    SETUP_PY="$(pwd)/scripts/setup.py"
    if [ -f "$SETUP_PY" ]; then
        cp "$SETUP_PY" "$CACHE"
    else
        echo "Warning: setup.py not found at $SETUP_PY"
    fi

    echo "Success! Now installing carla into your python package"
    conda activate opencda
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to activate conda environment 'opencda'"
    fi
    pip install -e "$CACHE"
fi

# Install OpenCOOD
echo "Installing OpenCOOD..."
conda activate opencda
if [ $? -ne 0 ]; then
    echo "Warning: Failed to activate conda environment 'opencda'"
fi
cd ./opencood
pip install -r ./requirements.txt
python ./setup.py develop
python ./opencood/utils/setup.py build_ext --inplace

echo "✅ Setup completed successfully!"
