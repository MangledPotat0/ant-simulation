################################################################################
#                                                                              #
#   Ant data processing suite deployment                                       #
#   Code written by Dawith Lim                                                 #
#                                                                              #
#   Version: 1.0.0.0.0.0                                                       #
#   First written on 2020/06/25                                                #
#   Last modified: 2020/06/25                                                  #
#                                                                              #
#   Description:                                                               #
#                                                                              #
#   Sets up the work environment for ant data extraction and processing on a   #
#   Linux or macOS system. For use on Windows, Windows Subsystem for Linux     #
#   (WSL) is required. Also, user computer must be connected to Purdue's       #
#   network to access source server on Eos.                                    #
#                                                                              #
################################################################################

#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  ## Install pyenv
  echo "OS Type: Linux"
  curl https://pyenv.run | bash
  source ~/.bashrc
  ## Set up virtualenv within pyenv
  pyenv install 3.7.4
  pyenv local 3.7.4
  pyenv global 3.7.4
  pyenv virtualenv mlempy
  pyenv local mlempy
  ## Set up python libraries
  pip install argparse, h5py, opencv-python, numpy, matplotlib, pandas, trackpy
  ## Set up ssh certificate and send it to Eos
  # ssh-keygen
  ## Set up directories
  mkdir -p ~/ant/data/{videos, density, montages, trajectories, plots}
  ## Clone git repository of codebase
  git clone admin@eos.physics.purdue.edu:volume1/codecase
  ## Add $PATH variable to codebase folder
  echo 'export PATH="$HOME/antworks/codebase:$PATH"' > ~/.bashrc
  source ~/.bashrc  
elif [[ "$OSTYPE" == "darwin"* ]]; then
  ## Install pyenv
  echo "OS Type: MacOs"
  curl https://pyenv.run | bash
  ## Set up virtualenv within pyenv
  pyenv install 3.7.4
  pyenv local 3.7.4
  pyenv global 3.7.4
  pyenv virtualenv mlempy
  pyenv local mlempy
  ## Set up python libraries
  pip install argparse, h5py, opencv-python, numpy, matplotlib, pandas, trackpy
  ## Set up ssh certificate and send it to Eos
  # ssh-keygen
  ## Set up directories
  mkdir -p ~/ant/data/{videos, density, montages, trajectories, plots}
  ## Clone git repository of codebase
  git clone admin@eos.physics.purdue.edu:volume1/codecase
  ## Add $PATH variable to codebase folder
  echo 'export PATH="$HOME/antworks/codebase:$PATH"' > ~/.profile
  source ~/.profile  
else
  echo "Unsupported OS Type. Use MacOS or Linux-based OS." 
fi
