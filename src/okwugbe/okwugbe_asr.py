import csv
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


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
        return waveform, utterance

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
