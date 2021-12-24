import sys
import os
import torch.nn as nn
import torchaudio
import torch
from cvutils import Validator


class HidePrintStatement: #This originated from with the need to hide the cvutils Validator statements  
    def __init__(self):
        super(HidePrintStatement, self).__init__()
        
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._stderr= sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr= open(os.devnull, 'w')
       
        return self._original_stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr=self._stderr


class process:
    def __init__(self):
        super(process, self).__init__()

    def data_processing(self, data, text_transform, data_type,n_feats,freq_mask,time_mask,common_voice):

        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_feats),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_feats)
        test_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_feats)

        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        audio_paths = []

        for waveform, utterance,audio_path in data:

            if common_voice['use_common_voice']==True:   

                with HidePrintStatement() as hp_:
                    try:
                        validator = Validator(common_voice['lang'])
                        utterance_validated = validator.validate(utterance)
                        if utterance_validated is not None:
                            utterance = utterance_validated
                    except Exception:
                        pass        
            
            if data_type == 'train':
                spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            elif data_type == 'valid':
                spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            elif data_type == 'test':
                spec = test_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                raise Exception("'data_type' should be train, valid or test.")
            spectrograms.append(spec)
            label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0] // 2)
            label_lengths.append(len(label))
            audio_paths.append(audio_path)

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths,audio_paths
