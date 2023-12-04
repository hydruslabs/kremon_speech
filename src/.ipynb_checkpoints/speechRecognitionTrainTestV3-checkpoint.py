#Author: David Bastien
#-------------------------------------TRAINING LOG--------------------------------------------------------
#Date: 15/06/2023
# Experiment 1: Logged as: 15_06_2023_11_16_45.pickle
# No model saved
# Trained for 1000 epochs with a learning rate: 0.01 Batch Size: 30 

# Experiment 2: TO DO
# Trained for 1000 epochs with a learning rate: 0.1 Batch Size: 30


import sys
sys.path.append('..')

from utils.performanceMetrics import cer
from utils.performanceMetrics import wer
from utils.misc import IterMeter
from utils.misc import GreedyDecoder

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
from datetime import datetime


import utils.transformations.textTransform as text_trans #????? 

import utils.networkConfigurations.SpeechRecognitionModel_0_0 as network

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = text_trans.TextTransform()




def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, utterance) in data:
        if data_type == 'train':
            spec = train_audio_transforms(torch.from_numpy(waveform).float()).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(torch.from_numpy(waveform).float()).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')

        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths








def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
    model.train()
    data_len = len(train_loader.dataset)
    train_loss = 0

    for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            train_loss += loss.item()/ len(train_loader)
            
            

           # if batch_idx % 1 == 0 or data_len == ((batch_idx + 1) * 5) :
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
             #       epoch, (batch_idx + 1) * len(spectrograms), data_len,
              #      100. * batch_idx / len(train_loader), loss.item()))
                
    
    return epoch,train_loss



def test(model, device, test_loader, criterion, epoch, iter_meter):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []

    with torch.no_grad():
            for i, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data 
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)


    return test_loss,avg_cer,avg_wer



def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)



#database_location = '../data/final/data_clean_batch_1.pickle'

#print("---------------------------------------------------------------------")
#print('Loading data from:'+database_location)
#print("---------------------------------------------------------------------")
#with open(database_location, 'rb') as handle:
#        b = pickle.load(handle)

pickle_list = ['cherlaine_augmented_0.pickle',
               'cherlaine_augmented_100.pickle'
]

b = []
for filename in pickle_list:
    print("Working with: "+filename)
    with open('/media/kremon-storage-space-2/dataset-03/'+filename, 'rb') as handle:
        b = b + pickle.load(handle)
        
random.shuffle(b)






kwargs = {'num_workers': 2, 'pin_memory': True} 

hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.35,
        "learning_rate": 5e-4,
        "batch_size": 5,
        "epochs": 100
    }

print(hparams)




train_2_loader = data.DataLoader(dataset=b[0:24000],
                                batch_size=hparams["batch_size"],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)


test_2_loader = data.DataLoader(dataset=b[24000:],
                                batch_size=5,
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

use_cuda = torch.cuda.is_available()

torch.manual_seed(7)

device = torch.device("cuda")





model = network.SpeechRecognitionModel(
            hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
            hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
            ).to(device)

print(model)

#Test different optimizers 
#optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
#optimizer = optim.SGD(model.parameters(), lr = hparams['learning_rate'], momentum = 0.3)

optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    

criterion = nn.CTCLoss(blank=28).to(device)


# Look into this learning rate scheduler seems odd?!
# anneal strategy????

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                                steps_per_epoch=int(len(train_2_loader)),
                                                epochs=hparams['epochs'],
                                                anneal_strategy='linear')

epochs = hparams['epochs']
iter_meter = IterMeter()
print(hparams)

epoch_list = []
train_loss_list = []
test_loss_list = []
avg_cer_list = []
avg_wer_list = []

now = datetime.now()
print("now =", now)
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

for epoch in range(1, epochs + 1):
        epoch,train_loss=train(model, device, train_2_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        test_loss,avg_cer,avg_wer=test(model, device, test_2_loader, criterion, epoch, iter_meter)
        print('Train Epoch: {}: Average loss: {:.4f} Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}'.format(epoch,train_loss,test_loss, avg_cer, avg_wer))
        
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        avg_cer_list.append(avg_cer)
        avg_wer_list.append(avg_wer)
        output = np.vstack((np.array(epoch_list),np.array(train_loss_list),np.array(test_loss_list),np.array(avg_cer_list),np.array(avg_wer_list))).T

        with open(dt_string+'.pickle', 'wb') as handle:
             pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

             
        if epoch % 5 == 0:
            print("-----------------------------------------------------")
            print("Saving Neural Network State"+"model_"+dt_string+"_"+str(epoch)+".pt")
            print("-----------------------------------------------------")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
                }, "/media/kremon-storage-space-2/models"+"model_"+dt_string+"_"+str(epoch)+".pt")