# Base of of Deep Speech 2 with some improvements including the addition of the attention mechanism
# Bonaventure Dossou

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)

        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first, with_attention=False, num_layers=1):
        super(BidirectionalGRU, self).__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.with_attention = with_attention

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, hidden = self.BiGRU(x)

        if self.with_attention:

            # Beginning of the attention part --- Bonaventure Dossou
            if self.batch_first:
                hidden = hidden.transpose(0, 1).contiguous()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            w1 = nn.Linear(x.size()[2], x.size()[2] // 2).to(device)
            w2 = nn.Linear(hidden.size()[2], hidden.size()[2]).to(device)

            v = nn.Linear(x.size()[2] // 2, x.size()[2]).to(device)

            x1 = w1(x).to(device)
            hidden2 = w2(hidden).to(device)

            del w1, w2
            torch.cuda.empty_cache()

            if hidden.size()[1] != x1.size()[1]:
                additional = np.full((hidden.size()[0], x1.size()[1] - hidden.size()[1], hidden.size()[2]), 0)
                hidden2 = torch.cat((hidden2, torch.tensor(additional, dtype=torch.float).to(device)), 1)
                del additional
                torch.cuda.empty_cache()

            m = nn.Tanh()
            score = v(m(x1 + hidden2)).to(device)  # compute attention scores

            del x1, hidden2, m, v
            torch.cuda.empty_cache()

            n = nn.Softmax()
            attention_weights = n(score)  # get attention weights

            del score, n
            torch.cuda.empty_cache()
            context_vector = attention_weights * x  # compute the attention vector
            x = torch.cat((context_vector, x), dim=-1)  # apply context vector to the input

            del context_vector
            torch.cuda.empty_cache()
            # End of the attention part - Bonaventure Dossou

            x = self.dropout(x)
        else:
            x = self.dropout(x)
        return x


class BidirectionalLSTM(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first, num_layers=1):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiLSTM(x)  # enc_output, (hidden_state, cell_state)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn, n_rnn, rnn_dim, n_class, n_feats, in_channels=1, out_channels=32, kernel=3,
                 stride=2, dropout=0.1, with_attention=False, n_rnn_layers=1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // 2

        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel, kernel), stride=(stride, stride),
                             padding=kernel // 2)

        # n residual cnn layers with filter size of out_channels
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(out_channels, out_channels, kernel=kernel, stride=stride // 2, dropout=dropout,
                        n_feats=n_feats)
            for _ in range(n_cnn)
        ])
        self.fully_connected = nn.Linear(n_feats * out_channels, rnn_dim)

        self.birnn_layers1 = nn.Sequential(*[
            BidirectionalLSTM(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                              hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0, num_layers=n_rnn_layers)
            for i in range(n_rnn)
        ])

        self.birnn_layers2 = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0, num_layers=n_rnn_layers)
            for i in range(n_rnn)
        ])

        self.birnn_layers2_attention = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim * 2 if i == 0 else rnn_dim * 4,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=True, num_layers=n_rnn_layers, with_attention=with_attention)
            for i in range(n_rnn)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

        self.classifier_attention = nn.Sequential(
            nn.Linear(rnn_dim * 4, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

        self.with_attention = with_attention

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        x = self.birnn_layers1(x).to(device)

        if self.with_attention:
            x = self.birnn_layers2_attention(x).to(device)
            x = self.classifier_attention(x)
        else:
            x = self.birnn_layers2(x).to(device)
            x = self.classifier(x)
        return x
