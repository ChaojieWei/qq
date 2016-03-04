#!/usr/bin/env bash

# Change to scrpt directory
cd "$(dirname "$0")"
echo "Change to script directory" 
# First install scipy (required by scikit-learn)
echo "First install scipy (required by scikit-learn)"
pip install scipy==0.16.0

# Install required packages from online repositories and
# from local directory for nolearn
echo "Install required packages from online repositories and from local directory for nolearn"
pip install -r cnn/requirements.txt 
pip install -e cnn/src/nolearn

# Change to original directory
echo "DONE. Change back to the original directory"
cd -
exit 0