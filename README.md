## Okwugbe
Minimalist Automatic Speech Recognition Library for African Languages

## Context
NB: This library follows our work [Okwugbé](https://arxiv.org/abs/2103.07762) on ASR for Fon and Igbo. Based on the architecture of the network described in our
paper, it aims at easing the training process of ASR for other languages.
The primary targets are African languages, but it supports other languages as well
## Parameters
| Parameter | Description | default | 
| --- | --- | --- |
| `rnn_dim` | RNN Dimension & Hidden Size | 512 |
| `num_layers` | Number of Layers | 1 |
| `n_cnn` | Number of CNN components | 5 |
| `n_rnn` | Number of RNN components | 3 |
| `n_feats` | Number of features for the ResCNN | 128 |
| `in_channels` | Number of input channels of the ResCNN | 1 |
| `out_channels` | Number of output channels of the ResCNN | 32 |
| `kernel` | Kernel Size for the ResCNN | 3 |
| `stride` | Stride Size for the ResCNN | 2 |
| `padding` | Padding Size for the ResCNN | 1 |
| `dropout` | Dropout (kept unique for all components) | 0.1 |
| `with_attention` | True to use attention mechanism, False else | False |
| `batch_multiplier` | Batch multiplier for Gradient Accumulation) | 1 (no Gradient Accumulation) |
| `grad_acc` | Gradient Accumulation Option | False |
| `model_path` | Path for the saved model | './okwugbe_model' |
| `characters_set` | Path to the .txt file containing unique characters | required |
| `validation_set` | Validation set size | 0.2 |
| `train_path` | Path to training set | required |
| `test_path` | Path to testing set | required |
| `learning_rate` | Learning rate | 3e-5 |
| `batch_size` | Batch Size | 3e-5 |
| `patience` | Early Stopping Patience | 20 |
| `epochs` | Training epochs | 500 |
| `optimizer` | Optimizer | 'adamw' |

## Usage
* Import the trainer instance
    - from train_eval import Train_Okwugbe 
        - train_path = '/path/to/training_file.csv'
        - test_path = '/path/to/testing_file.csv'
        - characters_set = '/path/to/character_set.txt'
    
    - /path/to/training_file.csv and /path/to/testing_file.csv are meant to be csv files with two columns:
    - the first one containing the full paths to audio wav files
    - the second one containing the textual transcription of audio contents

* Initialize the trainer instance
    - train = Train_Okwugbe(train_path, test_path, characters_set)

* Start the training
    - train.run()

## TODO (as of now)
* Add automatic building of character set

## Citation
Please cite our paper using the citation below if you use our work in anyway:

`
@article{2103.07762,
Author = {Bonaventure F. P. Dossou and Chris C. Emezue},
Title = {OkwuGbé: End-to-End Speech Recognition for Fon and Igbo},
Year = {2021},
Eprint = {arXiv:2103.07762},
Howpublished = {African NLP, EACL 2021}
}`
