#!/usr/bin/env bash
# Pass pip extra options as script arguments
XOPTS=''
if [[ "$#" > "0" ]]; then
    XOPTS="${@:1}"
fi
# Change to scrpt directory
cd "$(dirname "$0")"
echo "Change to script directory" 
# First install scipy (required by scikit-learn)
echo "First install scipy (required by scikit-learn)"
pip install ${XOPTS} scipy==0.16.0

# Install required packages from online repositories and
# from local directory for nolearn
echo "Install required packages from online repositories and from local directory for nolearn"
pip install ${XOPTS} -r cnn/requirements.txt 
pip install ${XOPTS} -e cnn/src/nolearn

# Check/install scipy again (required for some environments)
echo "First install scipy (required by scikit-learn)"
pip install ${XOPTS} scipy==0.16.0

# Change to original directory
echo "DONE. Change back to the original directory"
cd -
exit 0