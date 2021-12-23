import torch.nn as nn
import torchaudio
import torch


class process:
    def __init__(self):
        super(process, self).__init__()

    def data_processing(self, data, text_transform, data_type="train",n_feats=128,freq_mask=30,time_mask=100):

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
