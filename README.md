# Convolutional Neural network (CNN) for vibrational spectroscopy data
>This repository contains a Python implementantion of a CNN using the
Nolearn/Lasagne/Theano libraries. 
>You can re-run the experiments on the provided vibrational spectroscopy 
datasets or load the best network configuration for each dataset.
>TODO allow the user to provide extra datasets

# How to setup your system
>To run the cnn.py Python script inside the cnn directory you need
to follow the simple steps below.

# Use Virtualenv
>Before installing python dependencies I advise you to use virtualenv 
(https://virtualenv.readthedocs.org/en/latest/) to avoid to mess up 
with your sistem libraries.
>If you don't have virtualenv installed follow the installation guide
available on the virtualenv site.
>To setup a virtual enviroment you just need to run this command:

    virtualenv venv

>And then to use the virtual environment

    source venv/bin/activate

>To deactivate the environment simply run

    deactivate

# Install required Python packages
>For running this script you need to have an updated version of Pip
installed (See https://pip.pypa.io/en/stable/ for how to install it).

### Linux / Mac OS X systems
>In order to use my code you have just to run the setup.sh 
with the command from:

    bash setup.sh

### Windows sistems
>Here https://sites.google.com/site/pydatalog/python/pip-for-windows 
you will find a good starting point for using pip and virtualenv on 
Windows once you configure Python for Windows properly.
>There is a non tested setup.bat similar to the Linux one. Take it
as a descriptive file for the steps to follow to install
the required dependencies and source files.
    

# Running the script
>If nothing went wrong you can already run the main script file
that implements a Convolutional Neural Network and it is already
set up for loading the datasets considered in the paper.
>To run the Python script file from the main directory:

    python cnn/cnn.py
    
>To select the dataset you can use the option:

    --dataset=DATASET
>where possible values for DATASET are

    [extFTIR,extNIR,extRaman,TABLET_NIR,TABLET_Raman,
    extWINE,OIL,COFFEE,extSTRAWBERRY]
              
>To load the best trained network you have to use in addition
the option:

    --load_dataset

>If you also want to see the value of the main parameters of the network
you have to use also

    --short_print_network

>Instead if you want to train a new network you have to add only the 
option:

    --hyperparameter_optimization=SEARCH_METHOD
    
>where the possible search methods are

    [RANDOM,EXHAUSTIVE]
    
>After selecting the dataset to use with --dataset=DATASET_NAME
>For the full list of the script options use

    --help
