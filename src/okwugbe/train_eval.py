import argparse
import os
from model import SpeechRecognitionModel
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from metrics import Metrics
from decoder import Decoders
from texttransform import TextTransform
from okwugbe_asr import OkwugbeDataset,OkwugbeDatasetForCommonVoice
from process import process
from earlystopping import EarlyStopping
import colorama
import numpy as np
from commonvoice import generate_character_set

# init the colorama module
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW
RED = colorama.Fore.RED

process = process()
decoders = Decoders()
metrics = Metrics()


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def valid(model, device, test_loader, criterion, iter_meter, experiment, text_transform, epoch):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []

    print('Epoch {} ~ Validation'.format(epoch))

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = decoders.greedy_decoder(text_transform, output.transpose(0, 1), labels,
                                                                     label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(metrics.cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(metrics.wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    experiment['val_loss'].append((test_loss, iter_meter.get()))
    experiment['cer'].append((avg_cer, iter_meter.get()))
    experiment['wer'].append((avg_wer, iter_meter.get()))

    valid_text = 'Validation set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}'.format(test_loss,
                                                                                                       avg_cer,
                                                                                                       avg_wer)
    print(f"{GREEN}{valid_text}{RESET}")
    return avg_wer


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment, valid_loader,
          best_wer, model_path, text_transform, early_stopping, batch_multiplier=1, grad_acc=False):
    
    clipping_value = 5 # Default value
    model.train()
    data_len = len(train_loader.dataset)
    train_loss = 0
    batch_train_loss = 0
    print('Epoch {} ~ Training started'.format(epoch))
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        # used if grad accumulation is used
        # This is the idea of grad accumulation to overcome memory issue
        if grad_acc:
            print('Using Gradient Accumulation')
            train_loss += loss.item() / (len(train_loader) * batch_multiplier)
            if (batch_idx + 1) % batch_multiplier == 0:
                optimizer.step()
                scheduler.step()
                iter_meter.step()
                model.zero_grad()  # reset gradients
                optimizer.zero_grad()
                batch_train_loss += train_loss
                train_loss = 0
        else:
            train_loss += loss.item() / len(train_loader)

  
            optimizer.step()
            scheduler.step()
            iter_meter.step()

        if batch_idx % 5 == 0 or batch_idx == data_len:
            text = 'Train Epoch {}: [{}/{} ({:.0f}%)] - Loss: {:.6f}'.format(epoch, batch_idx * len(spectrograms),
                                                                             data_len,
                                                                             100. * batch_idx / len(train_loader),
                                                                             loss.item())
            print(f"{YELLOW}{text}{RESET}")

    experiment['loss'].append((train_loss, iter_meter.get()))
    val_wer = valid(model, device, valid_loader, criterion, iter_meter, experiment, text_transform, epoch)  # wer

    for i in range(
            1):  # Just to enable the 'break statement' - this will run once like a simple if/else statement
        early_stopping(val_wer, model, model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if val_wer < best_wer:
        best_wer = val_wer
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_wer': best_wer
        }, model_path)
    else:
        print("No improvement in validation according to WER")

    return best_wer


def test(model, device, test_loader, criterion, text_transform):
    print('\nInference/Testing\n')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = decoders.greedy_decoder(text_transform, output.transpose(0, 1), labels,
                                                                     label_lengths)

            for j in range(len(decoded_preds)):
                print("Decoding Speech's Content")
                print("Audio's Transcription: {}".format(decoded_preds[j]))
                test_cer.append(metrics.cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(metrics.wer(decoded_targets[j], decoded_preds[j]))
                current_prediction = "Decoded target: {}\nDecoded prediction: {}\n".format(decoded_targets[j],
                                                                                           decoded_preds[j])
                print(current_prediction)
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    test_label = 'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer,
                                                                                                   avg_wer)
    print(f"{RED}{test_label}{RESET}")


def main(model, train_path, test_path, validation_size, learning_rate, batch_size, epochs, experiment, cnn_layer,
         rnn_layer,
         model_path, rnn_dim, text_transform, batch_multiplier, grad_acc, n_class, n_feats, stride, dropout, optimizer,
         patience,common_voice):
    if grad_acc:
        batch_size = batch_size // batch_multiplier

    print('Characters set: {}'.format(text_transform.char_map))
    print('Characters set length: {}'.format(len(text_transform.char_map)))
    print('Number of classes: {}'.format(len(text_transform.char_map) + 1))

    hparams = {
        "n_cnn_layers": cnn_layer,
        "n_rnn_layers": rnn_layer,
        "rnn_dim": rnn_dim,
        "n_class": n_class,
        "n_feats": n_feats,
        "stride": stride,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    use_cuda = torch.cuda.is_available()
    
    #torch.manual_seed(7) #this should be optional
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Training with: {}".format(device))

    if common_voice['use_common_voice']==True:     
        train_dataset=OkwugbeDatasetForCommonVoice(common_voice['lang'],"train",validation_size)
        valid_dataset= OkwugbeDatasetForCommonVoice(common_voice['lang'],"valid",validation_size)
        test_dataset = OkwugbeDatasetForCommonVoice(common_voice['lang'],"test",validation_size)
     
    else:
        train_dataset = OkwugbeDataset(train_path, test_path, "train", validation_size) 
        valid_dataset = OkwugbeDataset(train_path, test_path, "valid", validation_size) 
        test_dataset = OkwugbeDataset(train_path, test_path, "test", validation_size) 
        


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=hparams['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: process.data_processing(x, text_transform, 'train'),
                                               **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=hparams['batch_size'],
                                               shuffle=False,
                                               collate_fn=lambda x: process.data_processing(x, text_transform, 'valid'),
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=hparams['batch_size'],
                                              shuffle=False,
                                              collate_fn=lambda x: process.data_processing(x, text_transform, 'test'),
                                              **kwargs)

    model = model.to(device)

    print(
        "Model with {} CNN components, {} RNN components, and Rnn_Dim == {}".format(cnn_layer, rnn_layer, rnn_dim))

    print("Model's Total Num Model Parameters: {}".format(
        sum([param.nelement() for param in model.parameters()])))

    optimizer_ = optim.AdamW(model.parameters(), hparams['learning_rate'])  # default optimizer

    if optimizer.lower() == 'adamw':
        optimizer_ = optim.AdamW(model.parameters(), hparams['learning_rate'])
    if optimizer.lower() == 'sgd':
        optimizer_ = optim.SGD(model.parameters(), hparams['learning_rate'])
    if optimizer.lower() == 'adam':
        optimizer_ = optim.Adam(model.parameters(), hparams['learning_rate'])

    if optimizer.lower() not in ['adamw', 'sgd', 'adam']:
        raise ValueError("Current optimizers supported: ['adamw', 'sgd', 'adam']")

    criterion = nn.CTCLoss(blank=n_class - 1, zero_infinity=True).to(device)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer_, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    iter_meter = IterMeter()
    best_wer = 1000

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_saved = checkpoint['epoch']
        best_wer = checkpoint['best_wer']
        print("Training resumed from epoch {} with best WER == {}".format(epoch_saved + 1, best_wer))
        early_stopping = EarlyStopping(patience, model_path, best_wer, best_wer)
        for epoch in range(epoch_saved + 1, epochs + 1):
            best_wer = train(model, device, train_loader, criterion, optimizer_, scheduler, epoch, iter_meter,
                             experiment, valid_loader, best_wer, model_path, text_transform, early_stopping,
                             batch_multiplier=batch_multiplier, grad_acc=grad_acc)
    else:
        early_stopping = EarlyStopping(patience, model_path, None, np.Inf)
        for epoch in range(1, epochs + 1):
            best_wer = train(model, device, train_loader, criterion, optimizer_, scheduler, epoch, iter_meter,
                             experiment, valid_loader, best_wer, model_path, text_transform, early_stopping,
                             batch_multiplier=batch_multiplier, grad_acc=grad_acc)

    print("Evaluating on Test data\n")
    test(model, device, test_loader, criterion, text_transform)


class Train_Okwugbe:
    def __init__(self, train_path=None, test_path=None,lang=None,use_common_voice=False, characters_set=None, n_cnn=5, n_rnn=3, rnn_dim=512, num_layers=1, n_feats=128,
                 in_channels=1, out_channels=32, kernel=3, stride=2, padding=1, dropout=0.1, with_attention=False,
                 batch_multiplier=1, grad_acc=False, model_path='okwugbe_model', learning_rate=3e-5, batch_size=80,
                 patience=20, epochs=500, optimizer='adamw', validation_size=0.2):
        if use_common_voice==True and lang==None:
            raise Exception(f'`lang` (language from Common Voice) must be specified if use_common_voice is set to True.')
        self.common_voice = {'use_common_voice':use_common_voice,'lang':lang.strip()}    

        if train_path==None and use_common_voice==False:
            raise Exception(f'`train_path` cannot be None')
        if test_path==None and use_common_voice==False:
            raise Exception(f'`test_path` cannot be None')
        self.train_path = train_path
        self.test_path = test_path
        self.characters_set = characters_set if characters_set is not None else generate_character_set(lang)
        self.n_cnn = n_cnn
        self.n_rnn = n_rnn
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.n_feats = n_feats
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.with_attention = with_attention
        self.batch_multiplier = batch_multiplier
        self.grad_acc = grad_acc
        self.model_path = model_path
        self.model_path = self.model_path+'_'+lang if lang!=None else self.model_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.epochs = epochs
        self.optimizer = optimizer
        self.validation_size = validation_size
        self.text_transform = TextTransform(self.characters_set)
        self.n_class = len(self.text_transform.char_map) + 1
        self.experiment = {'loss': [], 'val_loss': [], 'cer': [], 'wer': []}

    def run(self):
        asr_model = SpeechRecognitionModel(self.n_cnn, self.n_rnn, self.rnn_dim, self.n_class, self.n_feats,
                                           self.in_channels, self.out_channels, self.kernel,
                                           self.stride, self.dropout, self.with_attention, self.num_layers)

        main(asr_model, self.train_path, self.test_path, self.validation_size, self.learning_rate, self.batch_size,
             self.epochs, self.experiment, self.n_cnn, self.n_rnn, self.model_path, self.rnn_dim, self.text_transform,
             self.batch_multiplier, self.grad_acc, self.n_class, self.n_feats,
             self.stride, self.dropout, self.optimizer, self.patience,self.common_voice)
