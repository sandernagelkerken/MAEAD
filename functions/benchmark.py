import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from functions.combinations import combination_to_id
from functions.dataloader import load_torque
from sklearn import metrics
import time

class Vanilla_AE_shared(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations, n_hidden_layers, hidden_size:int = -1):
        super(Vanilla_AE_shared, self).__init__()                                     
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length                                  
        self.combinations = combinations
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        if self.n_hidden_layers == 0:
            self.fixed = nn.Sequential(nn.Linear(self.sequence_length, self.encoding_size),
                                       nn.ReLU(),
                                       nn.Linear(self.encoding_size, self.sequence_length))                                
        elif self.n_hidden_layers == 1:
            if hidden_size == -1:
                raise ValueError("Choose a hidden size")
            elif hidden_size < 1:
                raise ValueError("Choose a positive hidden size")
            self.fixed = nn.Sequential(nn.Linear(self.sequence_length, self.hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.encoding_size),
                                    nn.ReLU(),
                                    nn.Linear(self.encoding_size, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.sequence_length))
        else:
            raise NotImplementedError("Please choose between 0 or 1 hidden layers.")

    def forward(self, x):
        x = x[:,:self.sequence_length]
        x = self.fixed(x) 
        return x
    
class Vanilla_AE_separate(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations, n_hidden_layers, hidden_size:int = -1):
        super(Vanilla_AE_separate, self).__init__()                                     
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length                                  
        self.combinations = combinations
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        if self.n_hidden_layers == 0:                              
            self.variable = nn.ModuleDict()                                     
            for shifting_type in combinations:                                  
                for transformer_ID in combinations[shifting_type]:
                    combination_ID = combination_to_id(shifting_type,transformer_ID, self.combinations) 
                    self.variable[str(combination_ID)] = nn.Sequential(nn.Linear(self.sequence_length, self.encoding_size),
                                                        nn.ReLU(),
                                                        nn.Linear(self.encoding_size, self.sequence_length))
        elif self.n_hidden_layers == 1:
            if hidden_size == -1:
                raise ValueError("Choose a hidden size")
            elif hidden_size < 1:
                raise ValueError("Choose a positive hidden size")
            self.variable = nn.ModuleDict()                                         
            for shifting_type in combinations:                              
                for transformer_ID in combinations[shifting_type]:
                    combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                    self.variable[str(combination_ID)] = nn.Sequential(nn.Linear(self.sequence_length, self.hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.encoding_size),
                                    nn.ReLU(),
                                    nn.Linear(self.encoding_size, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.sequence_length)) 
        else:
            raise NotImplementedError("Please choose between 0 or 1 hidden layers.")

    def forward(self, x):
        x_id = x[:,self.sequence_length]
        sort = torch.argsort(x_id)
        reversesort = torch.argsort(sort)
        x = x[sort]
        y = []
        ids = torch.unique(x[:,self.sequence_length],sorted=False)
        for i in ids:
            x_i = x[x[:,self.sequence_length] == i]
            if x_i.shape[0] != 0:
                combination_ID = str(int(x_i[0,self.sequence_length].item()))
                x_i = x_i[:,:self.sequence_length]                           
                x_i = self.variable[combination_ID](x_i)                         
                y.append(x_i)
        x = torch.concat(y,0)
        x = x[reversesort]
        return x

class LSTM_AE_shared(nn.Module):
    def __init__(self, encoding_size, sequence_length, device):
        super(LSTM_AE_shared, self).__init__()                        
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length
        self.device = device
        self.encoder = nn.LSTM(input_size = 1,
                             hidden_size = self.encoding_size,
                             batch_first = True)
        self.decoderLSTM =  nn.LSTM(input_size = 1, hidden_size = self.encoding_size, batch_first = False)
        self.decoderLinear = nn.Linear(self.encoding_size, 1)

    def forward(self, x):
        x = x[:,:self.sequence_length]
        x = x.unsqueeze(2)
        transposed_x = torch.transpose(x, 0, 1)
        _, (h_t, c_t) = self.encoder(x)
        complete_output = torch.empty(0).to(self.device)
        last_output = self.decoderLinear(h_t)
        for step in range(self.sequence_length):
            last_output, (h_t, c_t) = self.decoderLSTM(last_output, (h_t, c_t))
            last_output = self.decoderLinear(last_output)
            complete_output = torch.cat((complete_output, last_output.squeeze(2)), dim = 0)
            if self.training:
                last_output = transposed_x[-(step+1),:,:].unsqueeze(0)
        x = torch.transpose(complete_output, 0, 1)
        x = torch.flip(x, [1])
        return x
    
class LSTM_AE_separate(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations, device, teacher_forcing:bool = True):
        super(LSTM_AE_separate, self).__init__()                        
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.device = device
        self.encoder = nn.ModuleDict()
        self.variableLSTMs = nn.ModuleDict()                                    #Create the dictionary for the variable decoder LSTM networks
        self.variableLinears = nn.ModuleDict()                                  #Create the dictionary for the variable linear layer for the reconstruction from LSTM outputs, as done in Malhotra et al. (2016)
        self.teacher_forcing = teacher_forcing                                  #Optionally, use teacher forcing
        for shifting_type in self.combinations:                                 
            for transformer_ID in self.combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type, transformer_ID, self.combinations)
                self.encoder[str(combination_ID)] = nn.LSTM(input_size = 1,
                                                            hidden_size = self.encoding_size,
                                                            batch_first = True)
                self.variableLSTMs[str(combination_ID)] = nn.LSTM(input_size = 1, hidden_size = self.encoding_size, batch_first = False)
                self.variableLinears[str(combination_ID)] = nn.Linear(self.encoding_size, 1)

    def forward(self, x):
        x_id = x[:,self.sequence_length]                                        #Retrieve the column containing the combination ids
        sort = torch.argsort(x_id)                                              #Create a sorting element that sorts the observations according to their combination id
        reversesort = torch.argsort(sort)                                       #Create an inverse sorting element to undo the sorting at the end of the forward pass
        x = x[sort]                                                             #Execute the sorting
        y = []                                                                  #Create a list to save the prediction results for each id
        ids = torch.unique(x[:,self.sequence_length], sorted = False)           #Create a list containing the ids in the batch
        for i in ids:                                                           #Iterate over all these ids
            x_i = x[x[:,self.sequence_length] == i]                             #Get the data from only this id
            if x_i.shape[0] != 0:
                combination_ID = str(int(i))                                    #Retrieve the combination ID
                x_i = x_i[:,:self.sequence_length]                              #Discard the column containing the ids
                x_i = x_i.unsqueeze(2)                                          #Create dimension 2: input size = 1
                if self.teacher_forcing:
                    transposed_x_i = torch.transpose(x_i, 0, 1)                 #Transpose the data such that it can be used as input for the batch_first = False decoder when using teacer forcing
                _ , (h_t, c_t) = self.encoder[combination_ID](x_i)                                #Run the data through the encoder and retrieve the encoding
                complete_output = torch.empty(0).to(self.device)                     #Initialize a tensor to save the decoder output
                last_output = self.variableLinears[combination_ID](h_t)         #Initialize 0-th output directly from the encoding
                for step in range(self.sequence_length):                        #Iterate one step at a time such that the output of the previous step can be used as input for the next
                    last_output, (h_t, c_t) = self.variableLSTMs[combination_ID](last_output, (h_t, c_t)) #Compute the output of LSTM cell for this step
                    last_output = self.variableLinears[combination_ID](last_output) #Compute the actual output from the LSTM output for this step
                    complete_output = torch.cat((complete_output, last_output.squeeze(2)), dim = 0) #Add this output to the tensor that saves the decoder output
                    if self.teacher_forcing:
                        if self.training:                                       #When teacher forcing is enabled during the training phase, use the actual result as input for the next step
                            last_output = transposed_x_i[-(step+1),:,:].unsqueeze(0)
                y.append(complete_output)                                       #Add the complete output for this id to the list
        x = torch.cat(y,1)                                                      #Concatenate the results for all ids
        x = torch.transpose(x, 0, 1)                                            #Transpose the result to bring it in the same format as the input
        x = x[reversesort]                                                      #Undo the sorting that was done at the start
        x = torch.flip(x, [1])                                                  #Reconstruct the sequence in reverse order, as done in Malhotra et al. (2016)
        return x

class GRU_AE_shared(nn.Module):
    def __init__(self, encoding_size, sequence_length, device):
        super(GRU_AE_shared, self).__init__()                        
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length
        self.device = device
        self.encoder = nn.GRU(input_size = 1,
                             hidden_size = self.encoding_size,
                             batch_first = True)
        self.decoderLSTM =  nn.GRU(input_size = 1, hidden_size = self.encoding_size, batch_first = False)
        self.decoderLinear = nn.Linear(self.encoding_size, 1)

    def forward(self, x):
        x = x[:,:self.sequence_length]
        x = x.unsqueeze(2)
        transposed_x = torch.transpose(x, 0, 1)
        _, h_t = self.encoder(x)
        complete_output = torch.empty(0).to(self.device)
        last_output = self.decoderLinear(h_t)
        for step in range(self.sequence_length):
            last_output, h_t = self.decoderLSTM(last_output, h_t)
            last_output = self.decoderLinear(last_output)
            complete_output = torch.cat((complete_output, last_output.squeeze(2)), dim = 0)
            if self.training:
                last_output = transposed_x[-(step+1),:,:].unsqueeze(0)
        x = torch.transpose(complete_output, 0, 1)
        x = torch.flip(x, [1])
        return x
    
class GRU_AE_separate(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations, device, teacher_forcing:bool = True):
        super(GRU_AE_separate, self).__init__()
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.device = device
        self.encoder = nn.ModuleDict()
        self.variableGRUs = nn.ModuleDict()                                     #Create the dictionary for the variable decoder GRU networks
        self.variableLinears = nn.ModuleDict()                                  #Create the dictionary for the variable linear layer for the reconstruction from GRU outputs, as done in Malhotra et al. (2016)
        self.teacher_forcing = teacher_forcing                                  #Optionally, use teacher forcing
        for shifting_type in combinations:                                      
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                self.encoder[str(combination_ID)] = nn.GRU(input_size = 1,                                     #Create the fixed encoder network
                                                            hidden_size = self.encoding_size,
                                                            batch_first = True)
                self.variableGRUs[str(combination_ID)] = nn.GRU(input_size = 1, hidden_size = self.encoding_size, batch_first = False)
                self.variableLinears[str(combination_ID)] = nn.Linear(self.encoding_size, 1)
            
    def forward(self, x):
        x_id = x[:,self.sequence_length]                                        #Retrieve the column containing the combination ids
        sort = torch.argsort(x_id)                                              #Create a sorting element that sorts the observations according to their combination id
        reversesort = torch.argsort(sort)                                       #Create an inverse sorting element to undo the sorting at the end of the forward pass
        x = x[sort]                                                             #Execute the sorting
        y=[]                                                                    #Create a list to save the prediction results for each id
        ids = torch.unique(x[:,self.sequence_length],sorted=False)              #Create a list containing the ids in the batch
        for i in ids:                                                           #Iterate over all these ids
            x_i = x[x[:,self.sequence_length] == i]                             #Get the data from only this id
            if x_i.shape[0] != 0:
                combination_ID = str(int(x_i[0,self.sequence_length].item()))   #Retrieve the combination ID
                x_i = x_i[:,:self.sequence_length]                              #Discard the column containing the ids
                x_i = x_i.unsqueeze(2)                                          #Create dimension 2: input size = 1
                if self.teacher_forcing:
                    transposed_x_i = torch.transpose(x_i, 0, 1)                 #Transpose the data such that it can be used as input for the batch_first = False decoder when using teacer forcing
                _, h_t = self.encoder[combination_ID](x_i)                                        #Run the data through the encoder and retrieve the encoding
                complete_output = torch.empty(0).to(self.device)                     #Initialize a tensor to save the decoder output
                last_output = self.variableLinears[combination_ID](h_t)         #Initialize 0-th output directly from the 
                for step in range(self.sequence_length):                        #Iterate one step at a time such that the output of the previous step can be used as input for the next
                    last_output, h_t = self.variableGRUs[combination_ID](last_output, h_t) #Compute the output of GRU cell for this step
                    last_output = self.variableLinears[combination_ID](last_output) #Compute the actual output from the GRU output for this step
                    complete_output = torch.cat((complete_output, last_output.squeeze(2)), dim = 0) #Add this output to the tensor that saves the decoder output
                    if self.teacher_forcing:
                        if self.training:                                       #When teacher forcing is enabled during the training phase, use the actual result as input for the next step
                            last_output = transposed_x_i[-(step+1),:,:].unsqueeze(0)
                y.append(complete_output)                                       #Add the complete output for this id to the list
        x = torch.cat(y,1)                                                      #Concatenate the results for all ids
        x = torch.transpose(x, 0, 1)                                            #Transpose the result to bring it in the same format as the input
        x = x[reversesort]                                                      #Undo the sorting that was done at the start
        x = torch.flip(x, [1])                                                  #Reconstruct the sequence in reverse order, as done in Malhotra et al. (2016)
        return x

class TCN_AE_shared(nn.Module):
    def __init__(self, sequence_length, n_channels:int, n_channels_reduced:int, n_encoding_channels:int, device, kernel_size:int = 5, pool_rate:int = 2, num_layers:int = 6):
        super(TCN_AE_shared, self).__init__()
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.n_channels_reduced = n_channels_reduced
        self.n_encoding_channels = n_encoding_channels
        self.pool_rate = pool_rate
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        self.encoder = nn.ModuleDict()
        for i in range(num_layers):
            if i == 0:
                self.encoder[str(i)] = nn.Sequential(nn.Conv1d(1, self.n_channels, self.kernel_size,padding = 'same', dilation = 2**i),
                                                     nn.ReLU(),
                                                     nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                     nn.ReLU())
            else:
                self.encoder[str(i)] = nn.Sequential(nn.Conv1d(self.n_channels_reduced, self.n_channels, self.kernel_size,padding = 'same', dilation = 2**i),
                                                     nn.ReLU(),
                                                     nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                     nn.ReLU())
        self.encoder["compressor"] = nn.Sequential(nn.Conv1d(self.n_channels_reduced*self.num_layers, self.n_encoding_channels, 1),
                                                   nn.AvgPool1d(self.pool_rate))
        self.decoder = nn.ModuleDict()
        for i in range(self.num_layers):
            if i == 0:
                self.decoder[str(i)] = nn.Sequential(nn.Conv1d(self.n_encoding_channels, self.n_channels, self.kernel_size, dilation=2**(self.num_layers-1-i), padding = 'same'),
                                                           nn.ReLU(),
                                                           nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                           nn.ReLU())
            else:
                self.decoder[str(i)] = nn.Sequential(nn.Conv1d(self.n_channels_reduced, self.n_channels, self.kernel_size, dilation=2**(self.num_layers-1-i), padding = 'same'),
                                                           nn.ReLU(),
                                                           nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                           nn.ReLU())
        self.decoder["compressor"] = nn.Conv1d(self.n_channels_reduced*self.num_layers, 1, 1)
        
    def forward(self, x):
        x = x[:,:self.sequence_length]
        x = x.unsqueeze(1)
        encoding = torch.empty(0).to(self.device)
        for j in range(self.num_layers):
            x = self.encoder[str(j)](x)
            encoding = torch.cat((encoding, x), dim = 1)
        encoding = self.encoder["compressor"](encoding)
        x = F.interpolate(encoding, self.sequence_length)
        decoding = torch.empty(0).to(self.device)
        for j in range(self.num_layers):
            x = self.decoder[str(j)](x)
            decoding = torch.cat((decoding, x), dim = 1)
        x = self.decoder["compressor"](decoding)
        x = x.squeeze(1)
        return x
    
class TCN_AE_separate(nn.Module):
    def __init__(self, sequence_length, combinations, n_channels:int, n_channels_reduced:int, n_encoding_channels:int, device, kernel_size:int = 5, pool_rate:int = 2, num_layers:int = 6):
        super(TCN_AE_separate, self).__init__()
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.n_channels = n_channels
        self.n_channels_reduced = n_channels_reduced
        self.n_encoding_channels = n_encoding_channels
        self.pool_rate = pool_rate
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        self.encoder = nn.ModuleDict()
        for shifting_type in combinations:                                      
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                self.encoder[str(combination_ID)] = nn.ModuleDict()
                for i in range(num_layers):
                    if i == 0:
                        self.encoder[str(combination_ID)][str(i)] = nn.Sequential(nn.Conv1d(1, self.n_channels, self.kernel_size,padding = 'same', dilation = 2**i),
                                                            nn.ReLU(),
                                                            nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                            nn.ReLU())
                    else:
                        self.encoder[str(combination_ID)][str(i)] = nn.Sequential(nn.Conv1d(self.n_channels_reduced, self.n_channels, self.kernel_size,padding = 'same', dilation = 2**i),
                                                            nn.ReLU(),
                                                            nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                            nn.ReLU())
                self.encoder[str(combination_ID)]["compressor"] = nn.Sequential(nn.Conv1d(self.n_channels_reduced*self.num_layers, self.n_encoding_channels, 1),
                                                        nn.AvgPool1d(self.pool_rate))
        self.decoder_variable = nn.ModuleDict()
        for shifting_type in combinations:                                      
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                self.decoder_variable[str(combination_ID)] = nn.ModuleDict()
                for i in range(self.num_layers):
                    if i == 0:
                        self.decoder_variable[str(combination_ID)][str(i)] = nn.Sequential(nn.Conv1d(self.n_encoding_channels, self.n_channels, self.kernel_size, dilation=2**(self.num_layers-1-i), padding = 'same'),
                                                                                        nn.ReLU(),
                                                                                        nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                                                        nn.ReLU())
                    else:
                        self.decoder_variable[str(combination_ID)][str(i)] = nn.Sequential(nn.Conv1d(self.n_channels_reduced, self.n_channels, self.kernel_size, dilation=2**(self.num_layers-1-i), padding = 'same'),
                                                                                        nn.ReLU(),
                                                                                        nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                                                        nn.ReLU())
                self.decoder_variable[str(combination_ID)]["compressor"] = nn.Conv1d(self.n_channels_reduced*self.num_layers, 1, 1)
        
    def forward(self, x):
        y = []
        ids = torch.unique(x[:,self.sequence_length],sorted=False)
        for i in ids:
            x_i = x[x[:,self.sequence_length] == i]
            if x_i.shape[0] != 0:
                combination_ID = str(int(x_i[0,self.sequence_length].item()))    #Read the combination ID of first element of the batch
                x_i = x_i[:,:self.sequence_length] 
                x_i = x_i.unsqueeze(1)
                encoding = torch.empty(0).to(self.device)
                for j in range(self.num_layers):
                    x_i = self.encoder[str(combination_ID)][str(j)](x_i)
                    encoding = torch.cat((encoding, x_i), dim = 1)
                encoding = self.encoder[str(combination_ID)]["compressor"](encoding)
                x_i = F.interpolate(encoding, self.sequence_length)
                decoding = torch.empty(0).to(self.device)
                for j in range(self.num_layers):
                    x_i = self.decoder_variable[str(combination_ID)][str(j)](x_i)
                    decoding = torch.cat((decoding, x_i), dim = 1)
                x_i = self.decoder_variable[str(combination_ID)]["compressor"](decoding)
                y.append(x_i)
        x = torch.concat(y,0)
        x = x.squeeze(1)
        return x
    
def create_Vanilla_AE_shared(encoding_size, sequence_length, combinations, n_hidden_layers, device, hidden_size:int = -1):
    model = Vanilla_AE_shared(encoding_size, sequence_length, combinations, n_hidden_layers, hidden_size).to(device)
    return model

def create_Vanilla_AE_separate(encoding_size, sequence_length, combinations, n_hidden_layers, device, hidden_size:int = -1):
    model = Vanilla_AE_separate(encoding_size, sequence_length, combinations, n_hidden_layers, hidden_size).to(device)
    return model

def create_TCN_AE_shared(sequence_length, n_channels, n_channels_reduced,
                    n_encoding_channels, device, kernel_size,
                    pool_rate, num_layers):
    model = TCN_AE_shared(sequence_length, n_channels, n_channels_reduced, 
                    n_encoding_channels, device, kernel_size, 
                    pool_rate, num_layers).to(device)
    return model

def create_TCN_AE_separate(sequence_length, combinations, n_channels, n_channels_reduced,
                    n_encoding_channels, device, kernel_size,
                    pool_rate, num_layers):
    model = TCN_AE_separate(sequence_length, combinations, n_channels, n_channels_reduced, 
                    n_encoding_channels, device, kernel_size, 
                    pool_rate, num_layers).to(device)
    return model

def create_LSTM_AE_shared(encoding_size, sequence_length, device):
    model = LSTM_AE_shared(encoding_size, sequence_length, device).to(device)
    return model

def create_LSTM_AE_separate(encoding_size, sequence_length, combinations, device):
    model = LSTM_AE_separate(encoding_size, sequence_length, combinations, device).to(device)
    return model

def create_GRU_AE_shared(encoding_size, sequence_length, device):
    model = GRU_AE_shared(encoding_size, sequence_length, device).to(device)
    return model

def create_GRU_AE_separate(encoding_size, sequence_length, combinations, device):
    model = GRU_AE_separate(encoding_size, sequence_length, combinations, device).to(device)
    return model

def create_DMOC(n_tasks:int, objective = 'soft-boundary', dimensions:int = 32):
    if n_tasks == 1:
        return deepsvdd1(objective, dimensions)
    elif n_tasks == 4:
        return deepsvdd4(objective, dimensions)
    elif n_tasks == 16:
        return deepsvdd16(objective, dimensions)
    else:
        raise NotImplementedError("Choose between 1, 4, and 16 tasks.")

class DMOC_CNN(nn.Module):
    def __init__(self, end_dim=32,pad=1):
        super(DMOC_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels  =1,
                out_channels = 16,
                kernel_size  = [1,5],
                stride = 1,
                padding = pad,
                ), 
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2,
                stride=2,
                ),
            nn.BatchNorm2d(16),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,[1,5],1,pad),  #(输入通道数，输出通道数，卷积核大小，步长，填充)
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,[1,5],1,pad),  #(输入通道数，输出通道数，卷积核大小，步长，填充)
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
        )
        self.fc1 = nn.Sequential(  #全连接层
            nn.Linear(2240,1280),
            nn.ReLU(),
            nn.BatchNorm1d(1280),
            nn.Linear(1280,640),
            nn.ReLU(),
            nn.BatchNorm1d(640),
            nn.Linear(640,end_dim),
           
            )
    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = x
        x = x.view(x.size(0),-1)
        output = self.fc1(x)
        return output

class deepsvdd4:
    def __init__(self, objective = 'soft-boundary', dimensions:int = 32):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.objective = objective

        self.R1 = torch.tensor(0, device=self.device)
        self.R2 = torch.tensor(0, device=self.device)
        self.R3 = torch.tensor(0, device=self.device)
        self.R4 = torch.tensor(0, device=self.device)
        self.lr_milestones = [20]

        self.weight_decay = 1e-4
        self.nu = 0.001
        self.warm_up_n_epochs = 10
        self.lr = 0.001
        self.batch_size = 128
        self.dim = dimensions
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.n_epochs = 0

        self.net = DMOC_CNN(self.dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_milestones, gamma=0.1)

    def get_data(self, data):

        data1 = data[data[:,:,:,-1] == 0].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data2 = data[data[:,:,:,-1] == 1].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data3 = data[data[:,:,:,-1] == 2].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data4 = data[data[:,:,:,-1] == 3].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]

        rand_index1 = np.random.choice(data1.shape[0], size=self.batch_size)
        rand_index2 = np.random.choice(data2.shape[0], size=self.batch_size)
        rand_index3 = np.random.choice(data3.shape[0], size=self.batch_size)
        rand_index4 = np.random.choice(data4.shape[0], size=self.batch_size)

        x_batch1 = data1[rand_index1]
        x_batch2 = data2[rand_index2]
        x_batch3 = data3[rand_index3]
        x_batch4 = data4[rand_index4]
        inputs = torch.cat([x_batch1, x_batch2, x_batch3, x_batch4], 0)
        return inputs
    

 # %%

    def svdd_model(self, core):
        outputs1 = core[:self.batch_size]
        outputs2 = core[self.batch_size:2*self.batch_size]
        outputs3 = core[2*self.batch_size:3*self.batch_size]
        outputs4 = core[3*self.batch_size:]

        self.c1 = self.init_center_c(outputs1)
        self.c2 = self.init_center_c(outputs2)
        self.c3 = self.init_center_c(outputs3)
        self.c4 = self.init_center_c(outputs4)

        self.dist1 = torch.sqrt(torch.sum((outputs1 - self.c1) ** 2, dim=1))
        self.dist2 = torch.sqrt(torch.sum((outputs2 - self.c2) ** 2, dim=1))
        self.dist3 = torch.sqrt(torch.sum((outputs3 - self.c3) ** 2, dim=1))
        self.dist4 = torch.sqrt(torch.sum((outputs4 - self.c4) ** 2, dim=1))

        if self.objective == 'soft-boundary':
            scores1 = self.dist1 - self.R1
            scores2 = self.dist2 - self.R2
            scores3 = self.dist3 - self.R3
            scores4 = self.dist4 - self.R4

            loss1 = self.R1 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores1), scores1))
            loss2 = self.R2 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores2), scores2))
            loss3 = self.R3 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores3), scores3))
            loss4 = self.R4 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores4), scores4))
        else:
            loss1 = torch.mean(self.dist1)
            loss2 = torch.mean(self.dist2)
            loss3 = torch.mean(self.dist3)
            loss4 = torch.mean(self.dist4)

        a = (loss2 + loss3 + loss4)/(3*(loss1 + loss2 + loss3 + loss4))
        b = (loss1 + loss3 + loss4)/(3*(loss1 + loss2 + loss3 + loss4))
        c = (loss1 + loss2 + loss4)/(3*(loss1 + loss2 + loss3 + loss4))
        d = (loss1 + loss2 + loss3)/(3*(loss1 + loss2 + loss3 + loss4))

        loss = a*loss1 + b*loss2 + c*loss3 + d*loss4
        return loss

    def MMD_loss(self, data):
        outputs1 = data[:self.batch_size]
        outputs2 = data[self.batch_size:2*self.batch_size]
        outputs3 = data[2*self.batch_size:3*self.batch_size]
        outputs4 = data[3*self.batch_size:]

        MMD_loss = torch.mean(torch.pow(outputs1-outputs2,2))\
            + torch.mean(torch.pow(outputs1-outputs3, 2))\
            + torch.mean(torch.pow(outputs2-outputs3, 2))\
            + torch.mean(torch.pow(outputs1-outputs4, 2))\
            + torch.mean(torch.pow(outputs2-outputs4, 2))\
            + torch.mean(torch.pow(outputs3-outputs4, 2))

        return MMD_loss
 # %% traing

    def train(self, data, epochs):
        initial_epochs = self.n_epochs
        self.n_epochs += epochs

        start_time = time.time()
        self.net.train()
        self.R_list = []
        self.C_list = []
        self.loss = []
        for epoch in range(initial_epochs, self.n_epochs):

            self.scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' %
                      float(self.scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for i in tqdm(range(4)):
                self.optimizer.zero_grad()
                inputs = self.get_data(data)
                inputs = inputs.to(self.device)
                self.outputs = self.net(inputs)

                MMD_loss = self.MMD_loss(self.outputs)

                svdd_loss = self.svdd_model(self.outputs)
                self.c_loss = torch.abs(torch.sum(self.c1-self.c2)
                                        + torch.sum(self.c1-self.c3)
                                        + torch.sum(self.c2-self.c3)
                                        + torch.sum(self.c1-self.c4)
                                        + torch.sum(self.c2-self.c4)
                                        + torch.sum(self.c3-self.c4))
                self.r_loss = self.R1 + self.R2 + self.R3 + self.R4

                loss = svdd_loss + 1000*MMD_loss + 10*self.c_loss + self.r_loss

                loss.backward()
                self.optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):

                    self.R1 = torch.tensor(self.get_radius(
                        self.dist1, self.nu), device=self.device)
                    self.R2 = torch.tensor(self.get_radius(
                        self.dist2, self.nu), device=self.device)
                    self.R3 = torch.tensor(self.get_radius(
                        self.dist3, self.nu), device=self.device)
                    self.R4 = torch.tensor(self.get_radius(
                        self.dist4, self.nu), device=self.device)
                self.R_list.append(self.R1)
                self.C_list.append(self.c1)

                loss_epoch += loss.item()
                n_batches += 1
            self.loss.append(loss_epoch)
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time

        return self.net, self.loss, self.C_list

  # %% test
     
    def test(self,data):
        self.net.eval()
        data_id = data.squeeze()[:,-2]
        sort = torch.argsort(data_id)
        reversesort = torch.argsort(sort)
        data = data[sort]
        all_scores = torch.empty(0).to(self.device)
        for i in range(4):
            data_i = data[data[:,:,:,-2] == i].unsqueeze(1).unsqueeze(1)[:,:,:,:-2]
            if data_i.shape[0] > 0:
                with torch.no_grad():
                    inputs = data_i.to(self.device)
                    outputs = self.net(inputs)
                    if i == 0:
                        dist = torch.sum((outputs - self.c1) ** 2, dim=1)
                    elif i == 1:
                        dist = torch.sum((outputs - self.c2) ** 2, dim=1)
                    elif i == 2:
                        dist = torch.sum((outputs - self.c3) ** 2, dim=1)
                    elif i == 3:
                        dist = torch.sum((outputs - self.c4) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        if i == 0:
                            scores = dist - self.R1 ** 2
                        elif i == 1:
                            scores = dist - self.R2 ** 2
                        elif i == 2:
                            scores = dist - self.R3 ** 2
                        elif i == 3:
                            scores = dist - self.R4 ** 2
                    else:
                        scores = dist
            all_scores = torch.concat((all_scores,scores))
        all_scores = all_scores[reversesort].cpu().data.numpy()
        self.test_scores = all_scores
        return self.test_scores
# %%

    def init_center_c(self, data, eps=0.01):

        n_samples = data.shape[0]
        c = torch.zeros(self.dim, device=self.device)

        c = torch.sum(data, dim=0)

        c = c / n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def init_center_c1(self, data, eps=0.01):

        c = torch.zeros(self.dim, device=self.device)

        c = torch.mean(torch.mean(torch.mean(data, dim=0), dim=1), dim=0)

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_radius(self, dist, nu):

        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

class deepsvdd1:
    def __init__(self, objective = 'soft-boundary', dimensions:int = 32):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.objective = objective

        self.R1 = torch.tensor(0, device=self.device)
        self.lr_milestones = [20]

        self.weight_decay = 1e-4
        self.nu = 0.001
        self.warm_up_n_epochs = 10
        self.lr = 0.001
        self.batch_size = 128
        self.dim = dimensions
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.n_epochs = 0

        self.net = DMOC_CNN(self.dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_milestones, gamma=0.1)

    def get_data(self, data):

        data1 = data[data[:,:,:,-1] == 0].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]

        rand_index1 = np.random.choice(data1.shape[0], size=self.batch_size)

        x_batch1 = data1[rand_index1]
        inputs = torch.cat([x_batch1], 0)
        return inputs
    

 # %%

    def svdd_model(self, core):
        outputs1 = core[:self.batch_size]

        self.c1 = self.init_center_c(outputs1)

        self.dist1 = torch.sqrt(torch.sum((outputs1 - self.c1) ** 2, dim=1))

        if self.objective == 'soft-boundary':
            scores1 = self.dist1 - self.R1

            loss1 = self.R1 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores1), scores1))
        else:
            loss1 = torch.mean(self.dist1)

        a = 1

        loss = a*loss1
        return loss

    def MMD_loss(self, data):
        outputs1 = data[:self.batch_size]

        MMD_loss = 0

        return MMD_loss
 # %% traing

    def train(self, data, epochs):
        initial_epochs = self.n_epochs
        self.n_epochs += epochs

        start_time = time.time()
        self.net.train()
        self.R_list = []
        self.C_list = []
        self.loss = []
        for epoch in range(initial_epochs, self.n_epochs):

            self.scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' %
                      float(self.scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for i in tqdm(range(4)):
                self.optimizer.zero_grad()
                inputs = self.get_data(data)
                inputs = inputs.to(self.device)
                self.outputs = self.net(inputs)

                MMD_loss = self.MMD_loss(self.outputs)

                svdd_loss = self.svdd_model(self.outputs)
                self.c_loss = 0
                self.r_loss = self.R1

                loss = svdd_loss + 1000*MMD_loss + 10*self.c_loss + self.r_loss

                loss.backward()
                self.optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):

                    self.R1 = torch.tensor(self.get_radius(
                        self.dist1, self.nu), device=self.device)

                self.R_list.append(self.R1)
                self.C_list.append(self.c1)

                loss_epoch += loss.item()
                n_batches += 1
            self.loss.append(loss_epoch)
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time

        return self.net, self.loss, self.C_list

  # %% test
     
    def test(self,data):
        self.net.eval()
        data_id = data.squeeze()[:,-2]
        sort = torch.argsort(data_id)
        reversesort = torch.argsort(sort)
        data = data[sort]
        all_scores = torch.empty(0).to(self.device)
        for i in range(4):
            data_i = data[data[:,:,:,-2] == i].unsqueeze(1).unsqueeze(1)[:,:,:,:-2]
            if data_i.shape[0] > 0:
                with torch.no_grad():
                    inputs = data_i.to(self.device)
                    outputs = self.net(inputs)
                    if i == 0:
                        dist = torch.sum((outputs - self.c1) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        if i == 0:
                            scores = dist - self.R1 ** 2
                    else:
                        scores = dist
                all_scores = torch.concat((all_scores,scores))
        all_scores = all_scores[reversesort].cpu().data.numpy()
        self.test_scores = all_scores
        return self.test_scores
# %%

    def init_center_c(self, data, eps=0.01):

        n_samples = data.shape[0]
        c = torch.zeros(self.dim, device=self.device)

        c = torch.sum(data, dim=0)

        c = c / n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def init_center_c1(self, data, eps=0.01):

        c = torch.zeros(self.dim, device=self.device)

        c = torch.mean(torch.mean(torch.mean(data, dim=0), dim=1), dim=0)

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_radius(self, dist, nu):

        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    
class deepsvdd16:
    def __init__(self, objective = 'soft-boundary', dimensions:int = 32):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.objective = objective

        self.R1 = torch.tensor(0, device=self.device)
        self.R2 = torch.tensor(0, device=self.device)
        self.R3 = torch.tensor(0, device=self.device)
        self.R4 = torch.tensor(0, device=self.device)
        self.R5 = torch.tensor(0, device=self.device)
        self.R6 = torch.tensor(0, device=self.device)
        self.R7 = torch.tensor(0, device=self.device)
        self.R8 = torch.tensor(0, device=self.device)
        self.R9 = torch.tensor(0, device=self.device)
        self.R10 = torch.tensor(0, device=self.device)
        self.R11 = torch.tensor(0, device=self.device)
        self.R12 = torch.tensor(0, device=self.device)
        self.R13 = torch.tensor(0, device=self.device)
        self.R14 = torch.tensor(0, device=self.device)
        self.R15 = torch.tensor(0, device=self.device)
        self.R16 = torch.tensor(0, device=self.device)
        self.lr_milestones = [20]

        self.weight_decay = 1e-4
        self.nu = 0.001
        self.warm_up_n_epochs = 10
        self.lr = 0.001
        self.batch_size = 128
        self.dim = dimensions
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.n_epochs = 0

        self.net = DMOC_CNN(self.dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_milestones, gamma=0.1)

    def get_data(self, data):

        data1 = data[data[:,:,:,-1] == 0].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data2 = data[data[:,:,:,-1] == 1].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data3 = data[data[:,:,:,-1] == 2].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data4 = data[data[:,:,:,-1] == 3].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data5 = data[data[:,:,:,-1] == 4].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data6 = data[data[:,:,:,-1] == 5].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data7 = data[data[:,:,:,-1] == 6].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data8 = data[data[:,:,:,-1] == 7].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data9 = data[data[:,:,:,-1] == 8].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data10 = data[data[:,:,:,-1] == 9].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data11 = data[data[:,:,:,-1] == 10].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data12 = data[data[:,:,:,-1] == 11].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data13 = data[data[:,:,:,-1] == 12].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data14 = data[data[:,:,:,-1] == 13].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data15 = data[data[:,:,:,-1] == 14].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]
        data16 = data[data[:,:,:,-1] == 15].unsqueeze(1).unsqueeze(1)[:,:,:,:-1]

        rand_index1 = np.random.choice(data1.shape[0], size=self.batch_size)
        rand_index2 = np.random.choice(data2.shape[0], size=self.batch_size)
        rand_index3 = np.random.choice(data3.shape[0], size=self.batch_size)
        rand_index4 = np.random.choice(data4.shape[0], size=self.batch_size)
        rand_index5 = np.random.choice(data5.shape[0], size=self.batch_size)
        rand_index6 = np.random.choice(data6.shape[0], size=self.batch_size)
        rand_index7 = np.random.choice(data7.shape[0], size=self.batch_size)
        rand_index8 = np.random.choice(data8.shape[0], size=self.batch_size)
        rand_index9 = np.random.choice(data9.shape[0], size=self.batch_size)
        rand_index10 = np.random.choice(data10.shape[0], size=self.batch_size)
        rand_index11 = np.random.choice(data11.shape[0], size=self.batch_size)
        rand_index12 = np.random.choice(data12.shape[0], size=self.batch_size)
        rand_index13 = np.random.choice(data13.shape[0], size=self.batch_size)
        rand_index14 = np.random.choice(data14.shape[0], size=self.batch_size)
        rand_index15 = np.random.choice(data15.shape[0], size=self.batch_size)
        rand_index16 = np.random.choice(data16.shape[0], size=self.batch_size)

        x_batch1 = data1[rand_index1]
        x_batch2 = data2[rand_index2]
        x_batch3 = data3[rand_index3]
        x_batch4 = data4[rand_index4]
        x_batch5 = data5[rand_index5]
        x_batch6 = data6[rand_index6]
        x_batch7 = data7[rand_index7]
        x_batch8 = data8[rand_index8]
        x_batch9 = data9[rand_index9]
        x_batch10 = data10[rand_index10]
        x_batch11 = data11[rand_index11]
        x_batch12 = data12[rand_index12]
        x_batch13 = data13[rand_index13]
        x_batch14 = data14[rand_index14]
        x_batch15 = data15[rand_index15]
        x_batch16 = data16[rand_index16]
        inputs = torch.cat([x_batch1, x_batch2, x_batch3, x_batch4, x_batch5, x_batch6, x_batch7, x_batch8, x_batch9, x_batch10, x_batch11, x_batch12, x_batch13, x_batch14, x_batch15, x_batch16], 0)
        return inputs
    

 # %%

    def svdd_model(self, core):
        outputs1 = core[:self.batch_size]
        outputs2 = core[self.batch_size:2*self.batch_size]
        outputs3 = core[2*self.batch_size:3*self.batch_size]
        outputs4 = core[3*self.batch_size:4*self.batch_size]
        outputs5 = core[4*self.batch_size:5*self.batch_size]
        outputs6 = core[5*self.batch_size:6*self.batch_size]
        outputs7 = core[6*self.batch_size:7*self.batch_size]
        outputs8 = core[7*self.batch_size:8*self.batch_size]
        outputs9 = core[8*self.batch_size:9*self.batch_size]
        outputs10 = core[9*self.batch_size:10*self.batch_size]
        outputs11 = core[10*self.batch_size:11*self.batch_size]
        outputs12 = core[11*self.batch_size:12*self.batch_size]
        outputs13 = core[12*self.batch_size:13*self.batch_size]
        outputs14 = core[13*self.batch_size:14*self.batch_size]
        outputs15 = core[14*self.batch_size:15*self.batch_size]
        outputs16 = core[15*self.batch_size:16*self.batch_size]

        self.c1 = self.init_center_c(outputs1)
        self.c2 = self.init_center_c(outputs2)
        self.c3 = self.init_center_c(outputs3)
        self.c4 = self.init_center_c(outputs4)
        self.c5 = self.init_center_c(outputs5)
        self.c6 = self.init_center_c(outputs6)
        self.c7 = self.init_center_c(outputs7)
        self.c8 = self.init_center_c(outputs8)
        self.c9 = self.init_center_c(outputs9)
        self.c10 = self.init_center_c(outputs10)
        self.c11 = self.init_center_c(outputs11)
        self.c12 = self.init_center_c(outputs12)
        self.c13 = self.init_center_c(outputs13)
        self.c14 = self.init_center_c(outputs14)
        self.c15 = self.init_center_c(outputs15)
        self.c16 = self.init_center_c(outputs16)
        self.clist = [self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.c10, self.c11, self.c12, self.c13, self.c14, self.c15, self.c16]

        self.dist1 = torch.sqrt(torch.sum((outputs1 - self.c1) ** 2, dim=1))
        self.dist2 = torch.sqrt(torch.sum((outputs2 - self.c2) ** 2, dim=1))
        self.dist3 = torch.sqrt(torch.sum((outputs3 - self.c3) ** 2, dim=1))
        self.dist4 = torch.sqrt(torch.sum((outputs4 - self.c4) ** 2, dim=1))
        self.dist5 = torch.sqrt(torch.sum((outputs5 - self.c5) ** 2, dim=1))
        self.dist6 = torch.sqrt(torch.sum((outputs6 - self.c6) ** 2, dim=1))
        self.dist7 = torch.sqrt(torch.sum((outputs7 - self.c7) ** 2, dim=1))
        self.dist8 = torch.sqrt(torch.sum((outputs8 - self.c8) ** 2, dim=1))
        self.dist9 = torch.sqrt(torch.sum((outputs9 - self.c9) ** 2, dim=1))
        self.dist10 = torch.sqrt(torch.sum((outputs10 - self.c10) ** 2, dim=1))
        self.dist11 = torch.sqrt(torch.sum((outputs11 - self.c11) ** 2, dim=1))
        self.dist12 = torch.sqrt(torch.sum((outputs12 - self.c12) ** 2, dim=1))
        self.dist13 = torch.sqrt(torch.sum((outputs13 - self.c13) ** 2, dim=1))
        self.dist14 = torch.sqrt(torch.sum((outputs14 - self.c14) ** 2, dim=1))
        self.dist15 = torch.sqrt(torch.sum((outputs15 - self.c15) ** 2, dim=1))
        self.dist16 = torch.sqrt(torch.sum((outputs16 - self.c16) ** 2, dim=1))

        if self.objective == 'soft-boundary':
            scores1 = self.dist1 - self.R1
            scores2 = self.dist2 - self.R2
            scores3 = self.dist3 - self.R3
            scores4 = self.dist4 - self.R4
            scores5 = self.dist5 - self.R5
            scores6 = self.dist6 - self.R6
            scores7 = self.dist7 - self.R7
            scores8 = self.dist8 - self.R8
            scores9 = self.dist9 - self.R9
            scores10 = self.dist10 - self.R10
            scores11 = self.dist11 - self.R11
            scores12 = self.dist12 - self.R12
            scores13 = self.dist13 - self.R13
            scores14 = self.dist14 - self.R14
            scores15 = self.dist15 - self.R15
            scores16 = self.dist16 - self.R16

            loss1 = self.R1 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores1), scores1))
            loss2 = self.R2 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores2), scores2))
            loss3 = self.R3 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores3), scores3))
            loss4 = self.R4 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores4), scores4))
            loss5 = self.R5 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores5), scores5))
            loss6 = self.R6 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores6), scores6))
            loss7 = self.R7 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores7), scores7))
            loss8 = self.R8 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores8), scores8))
            loss9 = self.R9 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores9), scores9))
            loss10 = self.R10 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores10), scores10))
            loss11 = self.R11 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores11), scores11))
            loss12 = self.R12 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores12), scores12))
            loss13 = self.R13 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores13), scores13))
            loss14 = self.R14 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores14), scores14))
            loss15 = self.R15 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores15), scores15))
            loss16 = self.R16 ** 2 + (1 / self.nu)\
                * torch.mean(torch.max(torch.zeros_like(scores16), scores16))
        else:
            loss1 = torch.mean(self.dist1)
            loss2 = torch.mean(self.dist2)
            loss3 = torch.mean(self.dist3)
            loss4 = torch.mean(self.dist4)
            loss5 = torch.mean(self.dist5)
            loss6 = torch.mean(self.dist6)
            loss7 = torch.mean(self.dist7)
            loss8 = torch.mean(self.dist8)
            loss9 = torch.mean(self.dist9)
            loss10 = torch.mean(self.dist10)
            loss11 = torch.mean(self.dist11)
            loss12 = torch.mean(self.dist12)
            loss13 = torch.mean(self.dist13)
            loss14 = torch.mean(self.dist14)
            loss15 = torch.mean(self.dist15)
            loss16 = torch.mean(self.dist16)

        a = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        b = (loss1 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        c = (loss2 + loss1 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        d = (loss2 + loss3 + loss1 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        e = (loss2 + loss3 + loss4 + loss1 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        f = (loss2 + loss3 + loss4 + loss5 + loss1+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        g = (loss2 + loss3 + loss4 + loss5 + loss6+ loss1 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        h = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss1 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        i = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss1 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        j = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss1 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        k = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss1 + loss12 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        l = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss1 + loss13 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        m = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss1 + loss14 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        n = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss1 + loss15 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        o = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss1 + loss16)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))
        p = (loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss1)\
            /(15*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16))

        loss = a*loss1 + b*loss2 + c*loss3 + d*loss4 + e*loss5 + f*loss6 + g*loss7 + h*loss8 + i*loss9 + j*loss10 + k*loss11 + l*loss12 + m*loss13 + n*loss14 + o*loss15 + p*loss16
        return loss

    def MMD_loss(self, data):
        outputs1 = data[:self.batch_size]
        outputs2 = data[self.batch_size:2*self.batch_size]
        outputs3 = data[2*self.batch_size:3*self.batch_size]
        outputs4 = data[3*self.batch_size:4*self.batch_size]
        outputs5 = data[4*self.batch_size:5*self.batch_size]
        outputs6 = data[5*self.batch_size:6*self.batch_size]
        outputs7 = data[6*self.batch_size:7*self.batch_size]
        outputs8 = data[7*self.batch_size:8*self.batch_size]
        outputs9 = data[8*self.batch_size:9*self.batch_size]
        outputs10 = data[9*self.batch_size:10*self.batch_size]
        outputs11 = data[10*self.batch_size:11*self.batch_size]
        outputs12 = data[11*self.batch_size:12*self.batch_size]
        outputs13 = data[12*self.batch_size:13*self.batch_size]
        outputs14 = data[13*self.batch_size:14*self.batch_size]
        outputs15 = data[14*self.batch_size:15*self.batch_size]
        outputs16 = data[15*self.batch_size:16*self.batch_size]
        outputslist = [outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10, outputs11, outputs12, outputs13, outputs14, outputs15, outputs16]

        # MMD_loss = torch.mean(torch.pow(outputs1-outputs2,2))\
        #     + torch.mean(torch.pow(outputs1-outputs3, 2))\
        #     + torch.mean(torch.pow(outputs2-outputs3, 2))\
        #     + torch.mean(torch.pow(outputs1-outputs4, 2))\
        #     + torch.mean(torch.pow(outputs2-outputs4, 2))\
        #     + torch.mean(torch.pow(outputs3-outputs4, 2))
        MMD_loss = 0
        for i in range(len(outputslist)):
            if i+1 <= len(outputslist):
                for j in range(i+1,len(outputslist)):
                    MMD_loss += torch.mean(torch.pow(outputslist[i]-outputslist[j],2))

        return MMD_loss
 # %% traing

    def train(self, data, epochs):
        initial_epochs = self.n_epochs
        self.n_epochs += epochs

        start_time = time.time()
        self.net.train()
        self.R_list = []
        self.C_list = []
        self.loss = []
        for epoch in range(initial_epochs, self.n_epochs):

            self.scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' %
                      float(self.scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for i in tqdm(range(4)):
                self.optimizer.zero_grad()
                inputs = self.get_data(data)
                inputs = inputs.to(self.device)
                self.outputs = self.net(inputs)

                MMD_loss = self.MMD_loss(self.outputs)

                svdd_loss = self.svdd_model(self.outputs)
                c_loss_temp = 0
                for i in range(len(self.clist)):
                    if i+1 <= len(self.clist):
                        for j in range(i+1, len(self.clist)):
                            c_loss_temp += torch.sum(self.clist[i]-self.clist[j])
                self.c_loss = torch.abs(c_loss_temp)
                self.r_loss = self.R1 + self.R2 + self.R3 + self.R4 + self.R5 + self.R6 + self.R7 + self.R8 + self.R9 + self.R10 + self.R11 + self.R12 + self.R13 + self.R14 + self.R15 + self.R16

                loss = svdd_loss + 1000*MMD_loss + 10*self.c_loss + self.r_loss

                loss.backward()
                self.optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):

                    self.R1 = torch.tensor(self.get_radius(
                        self.dist1, self.nu), device=self.device)
                    self.R2 = torch.tensor(self.get_radius(
                        self.dist2, self.nu), device=self.device)
                    self.R3 = torch.tensor(self.get_radius(
                        self.dist3, self.nu), device=self.device)
                    self.R4 = torch.tensor(self.get_radius(
                        self.dist4, self.nu), device=self.device)
                    self.R5 = torch.tensor(self.get_radius(
                        self.dist5, self.nu), device=self.device)
                    self.R6 = torch.tensor(self.get_radius(
                        self.dist6, self.nu), device=self.device)
                    self.R7 = torch.tensor(self.get_radius(
                        self.dist7, self.nu), device=self.device)
                    self.R8 = torch.tensor(self.get_radius(
                        self.dist8, self.nu), device=self.device)
                    self.R9 = torch.tensor(self.get_radius(
                        self.dist9, self.nu), device=self.device)
                    self.R10 = torch.tensor(self.get_radius(
                        self.dist10, self.nu), device=self.device)
                    self.R11 = torch.tensor(self.get_radius(
                        self.dist11, self.nu), device=self.device)
                    self.R12 = torch.tensor(self.get_radius(
                        self.dist12, self.nu), device=self.device)
                    self.R13 = torch.tensor(self.get_radius(
                        self.dist13, self.nu), device=self.device)
                    self.R14 = torch.tensor(self.get_radius(
                        self.dist14, self.nu), device=self.device)
                    self.R15 = torch.tensor(self.get_radius(
                        self.dist15, self.nu), device=self.device)
                    self.R16 = torch.tensor(self.get_radius(
                        self.dist16, self.nu), device=self.device)
                self.R_list.append(self.R1)
                self.C_list.append(self.c1)

                loss_epoch += loss.item()
                n_batches += 1
            self.loss.append(loss_epoch)
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time

        return self.net, self.loss, self.C_list

  # %% test
     
    def test(self,data):
        self.net.eval()
        data_id = data.squeeze()[:,-2]
        sort = torch.argsort(data_id)
        reversesort = torch.argsort(sort)
        data = data[sort]
        all_scores = torch.empty(0).to(self.device)
        for i in range(16):
            data_i = data[data[:,:,:,-2] == i].unsqueeze(1).unsqueeze(1)[:,:,:,:-2]
            if data_i.shape[0] > 0:
                with torch.no_grad():
                    inputs = data_i.to(self.device)
                    outputs = self.net(inputs)
                    if i == 0:
                        dist = torch.sum((outputs - self.c1) ** 2, dim=1)
                    elif i == 1:
                        dist = torch.sum((outputs - self.c2) ** 2, dim=1)
                    elif i == 2:
                        dist = torch.sum((outputs - self.c3) ** 2, dim=1)
                    elif i == 3:
                        dist = torch.sum((outputs - self.c4) ** 2, dim=1)
                    elif i == 4:
                        dist = torch.sum((outputs - self.c5) ** 2, dim=1)
                    elif i == 5:
                        dist = torch.sum((outputs - self.c6) ** 2, dim=1)
                    elif i == 6:
                        dist = torch.sum((outputs - self.c7) ** 2, dim=1)
                    elif i == 7:
                        dist = torch.sum((outputs - self.c8) ** 2, dim=1)
                    elif i == 8:
                        dist = torch.sum((outputs - self.c9) ** 2, dim=1)
                    elif i == 9:
                        dist = torch.sum((outputs - self.c10) ** 2, dim=1)
                    elif i == 10:
                        dist = torch.sum((outputs - self.c11) ** 2, dim=1)
                    elif i == 11:
                        dist = torch.sum((outputs - self.c12) ** 2, dim=1)
                    elif i == 12:
                        dist = torch.sum((outputs - self.c13) ** 2, dim=1)
                    elif i == 13:
                        dist = torch.sum((outputs - self.c14) ** 2, dim=1)
                    elif i == 14:
                        dist = torch.sum((outputs - self.c15) ** 2, dim=1)
                    elif i == 15:
                        dist = torch.sum((outputs - self.c16) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        if i == 0:
                            scores = dist - self.R1 ** 2
                        elif i == 1:
                            scores = dist - self.R2 ** 2
                        elif i == 2:
                            scores = dist - self.R3 ** 2
                        elif i == 3:
                            scores = dist - self.R4 ** 2
                        elif i == 4:
                            scores = dist - self.R5 ** 2
                        elif i == 5:
                            scores = dist - self.R6 ** 2
                        elif i == 6:
                            scores = dist - self.R7 ** 2
                        elif i == 7:
                            scores = dist - self.R8 ** 2
                        elif i == 8:
                            scores = dist - self.R9 ** 2
                        elif i == 9:
                            scores = dist - self.R10 ** 2
                        elif i == 10:
                            scores = dist - self.R11 ** 2
                        elif i == 11:
                            scores = dist - self.R12 ** 2
                        elif i == 12:
                            scores = dist - self.R13 ** 2
                        elif i == 13:
                            scores = dist - self.R14 ** 2
                        elif i == 14:
                            scores = dist - self.R15 ** 2
                        elif i == 15:
                            scores = dist - self.R16 ** 2
                    else:
                        scores = dist
            all_scores = torch.concat((all_scores,scores))
        all_scores = all_scores[reversesort].cpu().data.numpy()
        self.test_scores = all_scores
        return self.test_scores
# %%

    def init_center_c(self, data, eps=0.01):

        n_samples = data.shape[0]
        c = torch.zeros(self.dim, device=self.device)

        c = torch.sum(data, dim=0)

        c = c / n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def init_center_c1(self, data, eps=0.01):

        c = torch.zeros(self.dim, device=self.device)

        c = torch.mean(torch.mean(torch.mean(data, dim=0), dim=1), dim=0)

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_radius(self, dist, nu):

        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)