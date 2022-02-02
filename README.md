#Jupyter notebook for Data Scientist
Data scientist tool to create and run experiment with required model and training plan .


## Pre-requisite
1. Install Ananconda
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh 

bash Anaconda3-2020.11-Linux-x86_64.sh -b -p -y 

source ~/.bashrc 
```
2. Create an environment and activate it
```shell
conda create -n dsenvironemnt anaconda 

conda activate dsenvironemnt 
```
3. Install required package
```shell
pip install -r requirement.txt
```
4. Install Jupyter Notebook
```shell
conda install jupyter notebook

pip install jupyter
```
5. Move to Directory
```shell
cd federated-xray-datascientist
```
6. start Jupyter Notebook using any of below command
```shell
jupyter notebook
```
or 
```shell
~/.local/bin/jupyter-notebook 
```

# Testing Jupyter Notebook
There are some test models in this repo in test-models directory.

In order to test those copy the model file from test-models directory to parent directory.

# How to run an experiment
Follow this document guide to learn how to create and run an experiment.
