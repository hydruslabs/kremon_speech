import os
import pickle
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from pydub import AudioSegment, effects
import matplotlib.pyplot as plt



class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats, padding):
        super(ResidualCNN, self).__init__()
        self.cnn01 = nn.Conv2d(in_channels, out_channels, kernel, stride = 1, padding=padding)
        self.cnn02 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride = 1, padding=(0,0))
        self.batch_norm2 = nn.BatchNorm2d(n_feats//2)
        self.batch_norm3 = nn.BatchNorm2d(n_feats//2)
        self.batch_norm4 = nn.BatchNorm2d(n_feats//2)
        self.dropout02 = nn.Dropout(dropout)


    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.batch_norm2(x)
        #print("Input - Batch Norm 2")
        #print(x.size())
        x = self.cnn01(x)
        #print("Batch Norm 2 - CNN1")
        #print(x.size())
        x = self.batch_norm3(x)
        #print("CNN1 - Batch Norm3")
        #print(x.size())
        x = self.cnn02(x)
        #print("Batch Norm3 - CNN2")
        #print(x.size())
        x = self.batch_norm4(x)
        #print("CNN2 - Batch Norm4")
        #print(x.size())
        x = F.relu(x)
        #print("Batch Norm4 - ReLU")
        #print(x.size())
        #print("ReLU - Dropout")
        x = self.dropout02(x)
        #print("X and Residual")
        #print(x.size())
        #print(residual.size())
        x += residual

        return x # (batch, channel, feature, time)
    


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first, n_feats):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout04 = nn.Dropout(dropout)

    def forward(self, x):
       # print("Input GRU")
       # print(x.size())
        x = self.layer_norm(x)
       # print("Input GRU - Norm")
       # print(x.size())
        x = F.gelu(x)
        #print(" Norm - GeLU")
        #print(x.size())
        x, _ = self.BiGRU(x)
        #print("GeLU - BiGRU")
        #print(x.size())
        x = self.dropout04(x)
        #print("BiGRU - DropOut")
        #print(x.size())
        return x


class SpeechRecognitionModel(nn.Module):
    
    def __init__(self):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = 128
        rnn_dim = 1024
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, kernel_size=(41,11), stride=(2,2), padding=20)  # cnn for extracting heirachal features

        self.batch_norm1 = nn.BatchNorm2d(n_feats//2)
        self.dropout01 = nn.Dropout(0.1)

        

        self.rescnn_layer1 = ResidualCNN(32, 32, kernel=(7,3), stride=(1,1), dropout=0.2, n_feats=128//2, padding = (3,1) ) # LOOK INTO PARAMETERS!!!!!
        self.rescnn_layer2 = ResidualCNN(32, 32, kernel=(5,3), stride=(1,1), dropout=0.2, n_feats=128//2, padding = (2,1)) # LOOK INTO PARAMETERS!!!!!
        self.rescnn_layer3 = ResidualCNN(32, 32, kernel=(3,3), stride=(1,1), dropout=0.2, n_feats=128//2, padding = (1,1)) # LOOK INTO PARAMETERS!!!!!
        self.rescnn_layer4 = ResidualCNN(32, 32, kernel=(3,3), stride=(2,1), dropout=0.2, n_feats=128//2, padding = (1,1)) # LOOK INTO PARAMETERS!!!!!
        self.rescnn_layer5 = ResidualCNN(32, 32, kernel=(3,3), stride=(1,1), dropout=0.2, n_feats=128//2, padding = (1,1)) # LOOK INTO PARAMETERS!!!!!
            
         

        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layer1 = BidirectionalGRU(rnn_dim=rnn_dim,hidden_size=rnn_dim, dropout=0.3, batch_first=True, n_feats = n_feats) # LOOK INTO PARAMETERS!!!!!
        self.birnn_layer2 = BidirectionalGRU(rnn_dim=rnn_dim*2,hidden_size=rnn_dim, dropout=0.3, batch_first=False, n_feats = n_feats) # LOOK INTO PARAMETERS!!!!!
        self.birnn_layer3 = BidirectionalGRU(rnn_dim=rnn_dim*2,hidden_size=rnn_dim, dropout=0.3, batch_first=False, n_feats = n_feats) # LOOK INTO PARAMETERS!!!!!
        self.birnn_layer4 = BidirectionalGRU(rnn_dim=rnn_dim*2,hidden_size=rnn_dim, dropout=0.3, batch_first=False, n_feats = n_feats) # LOOK INTO PARAMETERS!!!!!

  

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(rnn_dim, 29)
        )
        

    def forward(self, x):
        #1 - Data Layer
        #    1.1 - CNN Standard
        #    1.2 - Batch Norm
        #    1.3 - ReLu Non-Linearity
        #    1.4 - Dropout [0.1]
        #-----------------------------------------------------
        x = self.cnn(x)
        #print("CNN Output")
        #print(x.size())
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout01(x)
        sizes = x.size()
        #print("We have a size here:")
        #print(x.size())

        #2 - Residual CNN 1
        #   2.1 - Batch Norm Layer
        #   2.2 - Channel-wise Conv Layer
        #   2.3 - Batch Norm Layer
        #   2.4 - 1 x 1 Conv Layer
        #   2.5 - Batch Norm Layer
        #   2.6 - ReLU Non-Linearity
        #   2.7 - Dropout [0.2]
        #-----------------------------------------------------
        #3 - Residual CNN 2
        #-----------------------------------------------------
        #7 - Residual CNN 3
        #-----------------------------------------------------
        #8 - Residual CNN 4
        #-----------------------------------------------------
        #9 - Residual CNN 5
        #-----------------------------------------------------

        x = self.rescnn_layer1(x)
        x = self.rescnn_layer2(x)
        x = self.rescnn_layer3(x)
        x = self.rescnn_layer4(x)
        x = self.rescnn_layer5(x)
        
        #------------------------------------------
        # WTF IS HAPENNING HERE!!!!!
        #------------------------------------------
        sizes = x.size()
       # print(x.size())
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        #print(x.size())
        x = self.fully_connected(x)
        #-------------------------------------------

        #10 - BiGRU 1
        #    10.1 - BiGRU
        #    10.2 - Batch Norm Layer
        #    10.3 - Dropout
        #-----------------------------------------------------
        #11 - BiGRU 2
        #-----------------------------------------------------
        #12 - BiGRU 3
        #-----------------------------------------------------
        #13 - BiGRU 4
        #-----------------------------------------------------
        x = self.birnn_layer1(x)
        x = self.birnn_layer2(x)
        x = self.birnn_layer3(x)
        x = self.birnn_layer4(x)

        #14 - Classifier
        #    14.1 - FC Layer
        #    14.2 - Batch Norm Layer
        #    14.3 - ReLU Non-Linearity
        #    14.4 - Dropout
        #    14.5 - FC Layer

        x = self.classifier(x)

        return x



