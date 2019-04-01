#!/bin/bash


########################
# Xavier Bresson, Sept 2018
########################



########################
# Configure the VM machine with Ubuntu 16.04 LTS, Tesla K80 GPU
########################

# Command line
# source ./install_python_gpu_script.sh



##################
# Install CUDA 9.0
##################

# echo "Checking for CUDA 9.0 and installing."
# if ! dpkg-query -W cuda-9-0; then
#   sudo curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
#   sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
#   sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
#   sudo apt-get update
#   sudo apt-get install cuda-9-0 -y
# fi
# # Enable persistence mode
# sudo nvidia-smi -pm 1



########################
# Install miniconda
########################

sudo apt-get update
sudo apt-get -y upgrade
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
./miniconda.sh -b
echo "PATH=~/miniconda3/bin:$PATH" >> ~/.bashrc
source ~/.bashrc



########################
# Install python libraries
########################

conda install -y python=3.6 numpy scipy ipython mkl jupyter seaborn pandas matplotlib scikit-learn networkx
conda install -y pytorch=0.4.1 torchvision cuda92 -c pytorch



########################
# Download project from github
########################

conda install -y git
git clone https://github.com/xbresson/CE7454_2018
cd CE7454_2018/codes/data/
python generate_all_data.py
cd



########################
# Notebooks configuration
########################

jupyter notebook --generate-config
echo "c = get_config()" >> .jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'" >> .jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> .jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> .jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.token='deeplearning'" >> .jupyter/jupyter_notebook_config.py



########################
# Run notebooks remotely
########################

# tmux
#tmux new -s deeplearning -d
#tmux send-keys "jupyter notebook" C-m
# use any web browser : http://xx.xx.xx.xx:8888
