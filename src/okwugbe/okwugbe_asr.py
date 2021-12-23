import csv
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class OkwugbeDataset(torch.utils.data.Dataset):
    """Create a Dataset for Okwugbe ASR.
    Args:
    data_type could be either 'test', 'train' or 'valid'
    """

    def __init__(self, train_path, test_path, datatype, validation_size=0.2):
        super(OkwugbeDataset, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.validation_size = validation_size

        self.train, self.validation, self.test = self.load_data()
        if datatype.lower() == 'train':
            self.data = self.get_data(self.train, datatype)

        if datatype.lower() == 'valid':
            self.data = self.get_data(self.validation, datatype)

        if datatype.lower() == 'test':
            self.data = self.get_data(self.test, datatype)

        """datatype could be either 'test', 'train' or 'valid' """

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        validation, testing = train_test_split(test, test_size=self.validation_size)
        return train, validation, testing

    def get_data(self, dataset, datatype):
        data = dataset.to_numpy()
        data = [data[i] for i in range(len(data)) if i != 0]
        print('{} set size: {}'.format(datatype.upper(), len(data)))
        return data

    def load_audio_item(self, d: list):
        utterance = d[1]
        wav_path = d[0]
        waveform, sample_rate = torchaudio.load(wav_path)
        return waveform, utterance,wav_path

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, utterance)``
        """
        fileid = self.data[n]
        return self.load_audio_item(fileid)

    def __len__(self) -> int:
        return len(self.data)


class OkwugbeDatasetForCommonVoice(torch.utils.data.Dataset): #If the user decides to use CommonVoice data instead
    """Create a Dataset for Okwugbe ASR using Common Voice data.
    Args:
    data_type could be either 'test', 'train' or 'valid'
    """

    def __init__(self, lang, datatype, validation_size=0.2):
        super(OkwugbeDatasetForCommonVoice, self).__init__()
        
        if datatype.lower().strip() not in ['train','test','valid']:
            raise Exception(f'`datatype` must be of type `train`,`test` or `valid`! ')        
        self.validation_size = validation_size
        self.resampler =  torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

        try:
            dataset = load_dataset("common_voice", lang)
        except Exception as data_error:
            raise Exception(f'Could not load this dataset from Common Voice. See error below: {data_error}')

        train = [d for d in dataset['train']]
        valid = [d for d in dataset['validation']]
        test = [d for d in dataset['test']]

        if len(valid)==0:
            #We will take the minimum of validation_size and 1% from train. This is because the language is very low-resource.
            valid_ratio = min(0.1,self.validation_size)
            if valid_ratio!=self.validation_size:
                print(f'Due to how small the dataset is, we will be using a validation size of {valid_ratio} instead of {self.validation_size} specified.  ')

            valid_size = int(valid_ratio*len(train))
            valid = train[:valid_size]
            train = train[valid_size:]
        
        
        if len(test)==0:
            #We will take the minimum of validation_size and 1% from train. This is because the language is very low-resource.
            test_ratio = min(0.1,self.validation_size)
            
            test_size = int(test_ratio*len(train))
            test = train[:test_size]
            train = train[test_size:]

        if datatype.lower().strip() == 'train':
            self.data = [d for d in train]
        if datatype.lower().strip() == 'valid':
            self.data = [d for d in valid]

        if datatype.lower().strip() == 'test':
            self.data = [d for d in test]

        """datatype could be either 'test', 'train' or 'valid' """

    

    def load_audio_item(self, d: list):
        utterance = d['sentence']
        wav_path = d['path']
        waveform, sample_rate = torchaudio.load(wav_path)

        #Common Voice is usually 48kHz so we resample to 16kHz
        waveform = self.resampler.forward(waveform.squeeze(0))
       
        return waveform, utterance,wav_path

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        """
        fileid = self.data[n]
        return self.load_audio_item(fileid)

    def __len__(self) -> int:
        return len(self.data)
