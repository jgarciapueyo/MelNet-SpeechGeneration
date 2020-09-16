# YAML configuration files
## Training YAML files
The training YAML file contains information needed to train the model:
- information about the dataset and how to transform the audio to melspectrograms,
- information about the architecture of the network (n_tiers, layers of each tier, ...),
- information about the dataset (name, path to the dataset, ...)
- information about the training (optimizer, epochs, ...)
- information about logging the training

### Fields of the YAML file
- name: (str) identifies the model to be trained and its parameters
- audio: parameters related with the datasets pipeline
    - sample_rate: (int): sample rate of the sample audio in Hz
    - spectrogram_type: (str) type of spectrogram it will represent
    - n_fft: (int) size of the Fast Fourier Transform
    - mel_channels: (int) number of mel channels when transforming to MelSpectrogram
    - hop_length: (int) length of hop between STFT windows
    - win_length: (int) window size
    - ref_level_db: (int) reference level (in dB) in spectrogram \[needed for the normalization of the dataset, similar to the one used [here](https://github.com/keithito/tacotron/blob/master/util/audio.py)]
    - min_level_db: (int) minimum level (in dB) in spectrogram \[needed for the normalization of the dataset, similar to the one used [here](https://github.com/keithito/tacotron/blob/master/util/audio.py)]
- network: parameters related with the structure of the network
    - n_tiers: (int)
    - layers: (List[int])
    - hidden_size: (int)
    - gmm_size: (int)
- data: information related to the folder where the datasets for training is found and additional  information depending on the dataset being used. The fields required change depending on the dataset being used.
- training:
    - optimizer: (str)
    - epochs: (int)
    - batch_size: (int)
    - lr: (float) learning rate
    - momentum: (float) needed depending on the optimizer
    - num_workers: (int) number of workers for loading the datasets using PyTorch dataloader
    - save_iterations: (int) iterations between saving weigths
    - dir_chkpt: (str) path to the folder used to save model weights
- logging:
    - dir_log_tensorboard: (str) path to the folder used to save the tensorboard logs
    - dir_log: (str) path to the folder used to save general logs
    - log_iterations: (int) iterations between saving logs
    
A template for a training YAML file can be found in [template_training.yml](template_training.yml).

## Synthesis YAML files
The synthesis YAML file contains information needed to generate new spectrograms:
- information about the path of the weights of the tiers being used
- information about the path used to store the output

NOTE: the synthesis YAML file has to be created with information according to the architecture of the MelNet model defined in the corresponding training YAML file. This means that if the model in the training file is defined to have 6 tiers, the synthesis YAML file has to define the checkpoint to six tiers.  
In addition, even though the tiers are trained separately, we can not mix tiers from different models (especially if hidden size is different).

### Fields of the YAML file
- name: (str) identifies the file to do synthesis
- checkpoints_path: (List[str]) path from project directory to the weights of the tiers of the model to perform synthesis. The tiers have to be ordered, from the first tier of the model being the first element in the list to the last tier of the model being the last item in the list.
- output_path: (str) path to save the results of synthesis

A template for a synthesis YAML file can be found in [template_synthesis.yml](template_synthesis.yml).
