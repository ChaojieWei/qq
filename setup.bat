::Change to scrpt directory
::DIR "$(dirname "$0")"
::ECHO "Change to script directory" 
::First install scipy (required by scikit-learn)
ECHO First install scipy (required by scikit-learn)
pip install scipy==0.16.0

::Install required packages from online repositories and
::from local directory for nolearn
ECHO Install required packages from online repositories and from local directory for nolearn
pip install -r cnn/requirements.txt 
pip install -e cnn/src/nolearn

::Change to original directory
::ECHO "DONE. Change back to the original directory"
::DIR -
