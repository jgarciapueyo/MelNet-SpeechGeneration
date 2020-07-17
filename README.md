# MelNet: PyTorch implementation

This project is a PyTorch implementation of the [MelNet paper](https://arxiv.org/abs/1906.01083) 
which aims at generating high-fidelity audio samples by using two-dimensional time-frequency 
representations (spectrograms) in conjunction with a highly expressive probabilistic model and
 multiscale generation procedure.
 
# Table of contents
1. Results
2. [Project Structure](#project-structure)
3. [Installation](#installation)
    1. [Installation with Ananconda](#installation-with-anaconda)
    2. Installation with Docker
4. [Usage](#usage)
5. [Notes](#notes)

## Project Structure
```
SpeechGeneration-MelNet
|
|-- data          <- original data used to train the model (you have to create it)
|
|-- logs             <- (you have to create it or it will be created automatically)
|   |-- general      <- logs for general training
|   `-- tensorboard  <- logs for displaying in tensorboard
|
|-- models
|   |-- chkpt     <- model weigths for different runs stored in pickle format. It stores also the
|   |                training parameters. (you have to create it or it'll be created automatically)
|   `-- params    <- description of the parameters to train and do speech synthesis according 
|                    to the paper and the dataset
|
|-- notebooks     <- Jupyter Notebooks explaining different parts of the data pipeline 
|                    or the model
|
|-- src                  <- source code for use in this project
|   |-- data             <- scripts to download and load the data
|   |-- dataprocessing   <- scripts to turn raw data into processed data to input to the model
|   |-- model            <- scripts of the model presented in the paper
|   `-- utils            <- scripts that are useful in the project
|
|-- environment.yml      <- file for reproducting the environment (created with anaconda)  
`-- Makefile             <- file with commands to run the project without effort
```
 
## Installation 
### Installation with Anaconda
0. Download and install [Anaconda](https://www.anaconda.com/)
1. Clone the [source code](https://github.com/jgarciapueyo/MelNet-SpeechGeneration) with git:
```
git clone https://github.com/jgarciapueyo/MelNet-SpeechGeneration
cd MelNet-SpeechGeneration
```
2. Prepare the environment with Anaconda
```
conda create --name melnet -f environment.yml
```

## Usage


## Notes
This project is part of the course [DD2465 Advanced, Individual Course in Computer Science](https://www.kth.se/student/kurser/kurs/DD2465?l=en)
during my studies at [KTH](https://www.kth.se/en).