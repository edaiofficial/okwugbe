## Okwugbe
Automatic Speech Recognition Library for (low-resource) African Languages


## Motivation
Our aim is to foster ASR for African languages by making the whole process--from dataset gathering and preprocessing to training--as easy as possible. This library follows our work [Okwugbé](https://arxiv.org/abs/2103.07762) on ASR for Fon and Igbo. Based on the architecture of the network described in our
paper, it aims at easing the training process of ASR for other languages.
The primary targets are African languages, but it supports other languages as well


## Usage
```pip install okwugbe```
```python
#Import the trainer instance
from train_eval import Train_Okwugbe 

train_path = '/path/to/training_file.csv'
test_path = '/path/to/testing_file.csv'
characters_set = '/path/to/character_set.txt'
 
"""
 /path/to/training_file.csv and /path/to/testing_file.csv are meant to be csv files with two columns:
    the first one containing the full paths to audio wav files
    the second one containing the textual transcription of audio contents
"""

#Initialize the trainer instance
train = Train_Okwugbe(train_path, test_path, characters_set)

#Start the training
train.run()
```
### Parameters
Here are the parameters for the package, as well as their default values.

The defualt valuse have been chosen so that you only have to make minimal changes to get a good ASR model going.
| Parameter | Description | default | 
| --- | --- | --- |
| `use_common_voice` | Whether or not to use common voice | False |
| `lang` | language to use from Common Voice. Must be specified if `use_common_voice` is set to True. | None |
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
| `batch_size` | Batch Size | 20 |
| `patience` | Early Stopping Patience | 20 |
| `epochs` | Training epochs | 500 |
| `optimizer` | Optimizer | 'adamw' |
|`freq_mask`|frequency masking (for speech augmentation)|30|
|`time_mask`| time masking (for speech augmentation) | 100 |
|`display_plot`| whether or not to plot metrics during training| True|

### Integration with Common Voicee
You easily train on [Common Voice](https://commonvoice.mozilla.org/en) data set with Okwugbe by specifying `use_common_voice=True` and setting `lang` to the language code of your choice. This language must be hosted on Common Voice.
```python
#Initialize the trainer instance
train = Train_Okwugbe(use_common_voice=True, lang='mn') # for mongolian

#Start the training
train.run()
```
Here is the list of our current supported languages in Common Voice.
|tt|en|de|fr|cy|br|cv|tr|ky|ga-IE|kab|ca|zh-TW|sl|it|nl|cnh|eo|et|fa|pt|eu|es|zh-CN|mn|sah|dv|rw|sv-SE|ru|id|ar|ta|ia|lv|ja|vot|ab|zh-HK|rm-sursilv|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|tatar|english|german|french|welsh|breton|chuvash|turkish|kyrgyz|irish|kabyle|catalan|taiwanese|slovenian|italian|dutch|hakha chin|esperanto|estonian|persian|portuguese|basque|spanish|chinese|mongolian|sakha|dhivehi|kinyarwanda|swedish|russian|indonesian|arabic|tamil|interlingua|latvian|japanese|votic|abkhaz|cantonese|romansh sursilvan|

## Tutorials
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bZxd7yBOHlqIJBBUUImh8vwF4Zn_A7a5?usp=sharing) on using OkwuGbe
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12XiQCuQzOr7lye2sFCvsn4Ch_DNevx4u?usp=sharing) on using OkwuGbe with Common Voice 

## ASR Data for African languages
Wondering where to find dataset for your African language? Here are some resources to check:
- [OpenSLR](https://www.openslr.org/resources.php)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
- [AfricaNLP-Public-Datasets](https://github.com/Andrews2017/africanlp-public-datasets)


## Debugging
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PSWHO_umQPN8L9KACAVjKtc-Pgq0HxIY?usp=sharing) is strictly for debugging!


## Citation
Please cite our paper using the citation below if you use our work in anyway:

```
@inproceedings{dossou-emezue-2021-okwugbe,
    title = "{O}kwu{G}b{\'e}: End-to-End Speech Recognition for {F}on and {I}gbo",
    author = "Dossou, Bonaventure F. P.  and
      Emezue, Chris Chinenye",
    booktitle = "Proceedings of the Fifth Workshop on Widening Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.winlp-1.1",
    pages = "1--4",
    abstract = "Language is a fundamental component of human communication. African low-resourced languages have recently been a major subject of research in machine translation, and other text-based areas of NLP. However, there is still very little comparable research in speech recognition for African languages. OkwuGb{\'e} is a step towards building speech recognition systems for African low-resourced languages. Using Fon and Igbo as our case study, we build two end-to-end deep neural network-based speech recognition models. We present a state-of-the-art automatic speech recognition (ASR) model for Fon, and a benchmark ASR model result for Igbo. Our findings serve both as a guide for future NLP research for Fon and Igbo in particular, and the creation of speech recognition models for other African low-resourced languages in general. The Fon and Igbo models source code have been made publicly available. Moreover, Okwugbe, a python library has been created to make easier the process of ASR model building and training.",
}```
