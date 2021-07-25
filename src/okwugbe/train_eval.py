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
from okwugbe_asr import OkwugbeDataset
from process import process
from earlystopping import EarlyStopping
import colorama
import numpy as np

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
parser = argparse.ArgumentParser()

# =========== Model's Arguments ======================================================
# ----------------- LSTM & GRU Parameters --------------------------------------------
parser.add_argument("--rnn_dim", help="RNN Dimension & Hidden Size", default=512, type=int)
parser.add_argument("--num_layers", help="Number of Layers", default=1, type=int)

# ============================== ASR Model Parameters ================================
parser.add_argument("--n_cnn", help="Number of CNN components", default=5, type=int)
parser.add_argument("--n_rnn", help="Number of RNN components", default=3, type=int)
parser.add_argument("--n_class", help="Number of characters + 1", default=60, type=int)
parser.add_argument("--n_feats", help="Number of features for the ResCNN", default=128, type=int)
parser.add_argument("--in_channels", help="Number of input channels of the ResCNN", default=1, type=int)
parser.add_argument("--out_channels", help="Number of output channels of the ResCNN", default=32, type=int)

parser.add_argument("--kernel", help="Kernel Size for the ResCNN", default=3, type=int)
parser.add_argument("--stride", help="Stride Size for the ResCNN", default=2, type=int)
parser.add_argument("--padding", help="Padding Size for the ResCNN", default=1, type=int)
parser.add_argument("--dropout", help="Dropout for all components", default=0.1, type=float)
parser.add_argument("--with_attention",
                    help="True to include attention mechanism, False else", default=False, type=bool)

parser.add_argument("--batch_multiplier", help="Batch multiplier for Gradient Accumulation", default=1, type=int)
parser.add_argument("--grad_acc", help="Gradient Accumulation Option", default=False, type=bool)
parser.add_argument("--model_path", help="Path for the saved model", default='okwugbe_model', type=str)
parser.add_argument("--characters_set", help="Path to the .txt file containing unique characters", required=True,
                    type=str)

parser.add_argument("--validation_set", help="Validation set size", default=0.2, type=float)
parser.add_argument("--train_path", help="Path to training set", required=True, type=str)
parser.add_argument("--test_path", help="Path to testing set", required=True, type=str)
parser.add_argument("--learning_rate", help="Learning rate", default=3e-5, type=float)
parser.add_argument("--batch_size", help="Batch Size", default=20, type=int)
parser.add_argument("--patience", help="Early Stopping Patience", default=20, type=int)
parser.add_argument("--epochs", help="Training epochs", default=500, type=int)
parser.add_argument("--optimizer", help="Optimizer", default='adamw', type=str)


# Usage
# python train_eval.py --train_path C:/Users/pancr/Downloads/asr_fon_data/train_.csv --test_path C:/Users/pancr/Downloads/asr_fon_data/test_.csv --n_class 60

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
          best_wer, model_path, text_transform, patience=20, batch_multiplier=1, grad_acc=False, is_resumed=False):
    if is_resumed:
        early_stopping = EarlyStopping(patience, model_path, best_wer, best_wer)
    else:
        early_stopping = EarlyStopping(patience, model_path, None, np.Inf)
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

        loss.backward()
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
         patience):
    if grad_acc:
        batch_size = batch_size // batch_multiplier

    print('Characters set: {}'.format(text_transform.char_map))
    print('Characters set length: {}'.format(len(text_transform.char_map)))

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
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Training with: {}".format(device))

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
        for epoch in range(epoch_saved + 1, epochs + 1):
            best_wer = train(model, device, train_loader, criterion, optimizer_, scheduler, epoch, iter_meter,
                             experiment, valid_loader, best_wer, model_path, text_transform, patience=patience,
                             batch_multiplier=batch_multiplier, grad_acc=grad_acc, is_resumed=True)
    else:
        for epoch in range(1, epochs + 1):
            best_wer = train(model, device, train_loader, criterion, optimizer_, scheduler, epoch, iter_meter,
                             experiment, valid_loader, best_wer, model_path, text_transform, patience=patience,
                             batch_multiplier=batch_multiplier, grad_acc=grad_acc)

    print("Evaluating on Test data\n")
    test(model, device, test_loader, criterion, text_transform)


def run(args):
    n_cnn, n_rnn, rnn_dim, n_class = args.n_cnn, args.n_rnn, args.rnn_dim, args.n_class
    n_feats = args.n_feats
    in_channels, out_channels, kernel = args.in_channels, args.out_channels, args.kernel
    stride, padding = args.stride, args.padding
    dropout, with_attention, num_layers = args.dropout, args.with_attention, args.num_layers

    asr_model = SpeechRecognitionModel(n_cnn, n_rnn, rnn_dim, n_class, n_feats, in_channels, out_channels, kernel,
                                       stride, dropout, with_attention, num_layers)

    batch_multiplier, grad_acc, model_path, path_char_sets = args.batch_multiplier, args.grad_acc, args.model_path, args.characters_set

    validation_size, train_path, test_path, learning_rate = args.validation_set, args.train_path, args.test_path, args.learning_rate
    batch_size, epochs, optimizer, patience = args.batch_size, args.epochs, args.optimizer, args.patience

    text_transform = TextTransform(path_char_sets)
    experiment = {'loss': [], 'val_loss': [], 'cer': [], 'wer': []}

    main(asr_model, train_path, test_path, validation_size, learning_rate, batch_size, epochs, experiment, n_cnn, n_rnn,
         model_path, rnn_dim, text_transform, batch_multiplier, grad_acc, n_class, n_feats, stride, dropout, optimizer,
         patience)


if __name__ == '__main__':
    args_ = parser.parse_args()
    run(args_)
