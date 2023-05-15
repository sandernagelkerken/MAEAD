#Version 2, improved and GPU-enabled
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, Sampler, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from functions.combinations import combination_to_id
from functions.dataloader import load_torque
from sklearn import metrics

class TorqueDataset(Dataset):
    def __init__(self, cols, combinations, split:bool = False, train:bool = True, ratio:float = 0.8, path:str = "C:\LocalData\Data\Raw_data"):
        self.index = 0                                                          
        sub_df_torques = []
        self.combination_IDs_indices = {}                                       #Initialize a disctionary in which all the indices of a combination ID are kept, this is later used in creating a batch sampler
        for shifting_type in combinations:                                      #Iterate over all shifting type and transformer ID combinations
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type, transformer_ID, combinations) #Retrieve the combination_ID
                sub_df_torque = load_torque(shifting_type, transformer_ID, path, True)[cols] #Load the data for this combinations, set NA values to 0, and keep only the columns in cols
                if split:
                    if train:
                        sub_df_torque = sub_df_torque.iloc[:int(len(sub_df_torque)*ratio)]
                    else:
                        sub_df_torque = sub_df_torque.iloc[int(len(sub_df_torque)*ratio):]
                sub_df_torque["combination_ID"] = [combination_ID]*len(sub_df_torque) 
                sub_df_torques.append(sub_df_torque)                            #Add the combination ID to each row
                self.combination_IDs_indices[str(combination_ID)] = [int(x) for x in np.arange(self.index, self.index + len(sub_df_torque),1)] #Add the indices for each combination ID to the dictionary
                self.index += len(sub_df_torque)
        self.df_torque = pd.concat(sub_df_torques)                              #Concatenate all dataframes to obtain the complete dataset including combination IDs for each row

    def __len__(self):                                                          #Define a __len__ function (required)
        return len(self.df_torque)

    def __getitem__(self, idx):                                                 #Define a __getitem__ function (required)
        return torch.Tensor(self.df_torque.iloc[idx])                           #Transform the pd.DataFrame rows to torch tensors when an item is retrieved using the __getitem__ method (by using dataset[i])

    def get_IDs_indices(self):                                                  #Define a function for easy retrieval of the kept dictionary, will be used in the batch sampler
        return self.combination_IDs_indices

class Batch_IDs_together(Sampler):
    def __init__(self, dataset:TorqueDataset, batch_size:int, max_batches_per_ID:int, complete_batches_only:bool = False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.complete_batches_only = complete_batches_only
        self.IDs_indices = dataset.get_IDs_indices()
        self.max_batches_per_ID = max_batches_per_ID

    def __iter__(self):
        all_batches = []                                                        #Keep a list with all batches
        for ID in self.IDs_indices:                                             #Iterate over all combination IDs in the dataset    
            indices = self.IDs_indices[ID]                                      #Retrive all the indices corresponding to entries originating from the combination ID
            shuffle(indices)                                                    #Randomly shuffle these indices
            if self.complete_batches_only:
                to_remove = len(indices) % self.batch_size
                indices = indices[:-to_remove]
            batched_indices = torch.split(torch.tensor(indices), self.batch_size) #Split the shuffled indices up in batches
            if len(batched_indices) > self.max_batches_per_ID:
                batched_indices = batched_indices[:self.max_batches_per_ID]     #Take the first max_batches_per_ID batches only
            all_batches += batched_indices                                      #Add these batches to the list with all the batches
        all_batches = [batch.tolist() for batch in all_batches]                 #Transfrom these batches from tensors to lists
        shuffle(all_batches)                                                    #Shuffle all the batches
        return iter(all_batches)

    def __len__(self):                                                          #Compute the number of batches without computing the actual batches themselves
        return int(sum([np.ceil(len(i)/self.batch_size) for i in self.dataset.get_IDs_indices().values()])) 

class Batch_mixed_IDs(Sampler):
    def __init__(self, dataset, batch_size_per_ID:int):
        self.dataset = dataset
        self.IDs_indices = dataset.get_IDs_indices()
        self.batch_size_per_ID = batch_size_per_ID

    def __iter__(self):
        all_batches = []                                                        #Keep a list with all batches
        ID_batches = {}                                                         #Keep a dictrionary with batch components of each ID
        smallest_ID_len = 1_000_000_000                                         
        for id in self.IDs_indices:                                             #Iterate over all combination IDs in the dataset    
            indices = self.IDs_indices[id]                                      #Retrive all the indices corresponding to entries originating from the combination ID
            if len(indices) < smallest_ID_len:
                smallest_ID_len = len(indices)                                  
            shuffle(indices)                                                    #Randomly shuffle these indices
            batched_indices = torch.split(torch.tensor(indices), self.batch_size_per_ID) #Split the shuffled indices up in batch components
            ID_batches[id] = batched_indices                                    #Add these batch components to the disctionary
        for i in range(int(np.floor(smallest_ID_len/self.batch_size_per_ID))):  #Iterate until we reached the amount of complete batches we can make
            batch = torch.cat([ID_batches[id][i] for id in self.IDs_indices])        #Add the batch components together
            all_batches.append(batch)                                           #Add the batch tensor to all batches
        all_batches = [batch.tolist() for batch in all_batches]                 #Transform the tensors into lists
        return iter(all_batches)                                                #Return the generator

class Vanilla_MAE(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations, n_hidden_layers, hidden_size:int = -1, n_variable_layers:int = -1):
        super(Vanilla_MAE, self).__init__()                                     
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length                                  
        self.combinations = combinations
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.n_variable_layers = n_variable_layers
        if self.n_hidden_layers == 0:
            self.fixed = nn.Sequential(nn.Linear(self.sequence_length, self.encoding_size),
                                       nn.ReLU())                               
            self.variable = nn.ModuleDict()                                     
            for shifting_type in combinations:                                  
                for transformer_ID in combinations[shifting_type]:
                    combination_ID = combination_to_id(shifting_type,transformer_ID, self.combinations) 
                    self.variable[str(combination_ID)] = nn.Linear(self.encoding_size, self.sequence_length)   
        elif self.n_hidden_layers == 1:
            if hidden_size == -1:
                raise ValueError("Choose a hidden size")
            elif hidden_size < 1:
                raise ValueError("Choose a positive hidden size")
            if self.n_variable_layers == 1:
                self.fixed = nn.Sequential(nn.Linear(self.sequence_length, self.hidden_size), 
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.encoding_size),
                                        nn.ReLU(),
                                        nn.Linear(self.encoding_size, self.hidden_size),
                                        nn.ReLU())
                self.variable = nn.ModuleDict()                                         
                for shifting_type in combinations:                              
                    for transformer_ID in combinations[shifting_type]:
                        combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                        self.variable[str(combination_ID)] = nn.Linear(self.hidden_size, self.sequence_length)
            elif self.n_variable_layers == 2:
                self.fixed = nn.Sequential(nn.Linear(self.sequence_length, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.encoding_size),
                                        nn.ReLU())
                self.variable = nn.ModuleDict()                                     
                for shifting_type in combinations:                                  
                    for transformer_ID in combinations[shifting_type]:
                        combination_ID = combination_to_id(shifting_type,transformer_ID, combinations)
                        self.variable[str(combination_ID)] = nn.Sequential(nn.Linear(self.encoding_size, self.hidden_size),
                                                                    nn.ReLU(),
                                                                    nn.Linear(self.hidden_size, self.sequence_length)) 
            else:
                raise NotImplementedError("Please choose between 1 or 2 variable layers.")
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
                x_i = self.fixed(x_i)                                  
                x_i = self.variable[combination_ID](x_i)                         
                y.append(x_i)
        x = torch.concat(y,0)
        x = x[reversesort]
        return x

class TCN_MAE(nn.Module):
    def __init__(self, sequence_length, combinations, n_channels:int, n_channels_reduced:int, n_encoding_channels:int, n_variable_layers:int, device, kernel_size:int = 5, pool_rate:int = 2, num_layers:int = 6):
        super(TCN_MAE, self).__init__()
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.n_channels = n_channels
        self.n_channels_reduced = n_channels_reduced
        self.n_encoding_channels = n_encoding_channels
        self.pool_rate = pool_rate
        self.n_variable_layers = n_variable_layers
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
        self.decoder_fixed = nn.ModuleDict()
        for i in range(self.num_layers - self.n_variable_layers):
            if i == 0:
                self.decoder_fixed[str(i)] = nn.Sequential(nn.Conv1d(self.n_encoding_channels, self.n_channels, self.kernel_size, dilation=2**(self.num_layers-1-i), padding = 'same'),
                                                           nn.ReLU(),
                                                           nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                           nn.ReLU())
            else:
                self.decoder_fixed[str(i)] = nn.Sequential(nn.Conv1d(self.n_channels_reduced, self.n_channels, self.kernel_size, dilation=2**(self.num_layers-1-i), padding = 'same'),
                                                           nn.ReLU(),
                                                           nn.Conv1d(self.n_channels, self.n_channels_reduced, 1),
                                                           nn.ReLU())
        self.decoder_variable = nn.ModuleDict()
        for shifting_type in combinations:                                      
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                self.decoder_variable[str(combination_ID)] = nn.ModuleDict()
                for i in range(self.num_layers - self.n_variable_layers, self.num_layers):
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
                    x_i = self.encoder[str(j)](x_i)
                    encoding = torch.cat((encoding, x_i), dim = 1)
                encoding = self.encoder["compressor"](encoding)
                x_i = F.interpolate(encoding, self.sequence_length)
                decoding = torch.empty(0).to(self.device)
                for j in range(self.num_layers - self.n_variable_layers):
                    x_i = self.decoder_fixed[str(j)](x_i)
                    decoding = torch.cat((decoding, x_i), dim = 1)
                for j in range(self.num_layers - self.n_variable_layers, self.num_layers):
                    x_i = self.decoder_variable[str(combination_ID)][str(j)](x_i)
                    decoding = torch.cat((decoding, x_i), dim = 1)
                x_i = self.decoder_variable[str(combination_ID)]["compressor"](decoding)
                y.append(x_i)
        x = torch.concat(y,0)
        x = x.squeeze(1)
        return x

class LSTM_MAE(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations, device, teacher_forcing:bool = True):
        super(LSTM_MAE, self).__init__()                        
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.device = device
        self.fixed = nn.LSTM(input_size = 1,                                    #Create the fixed encoder network
                             hidden_size = self.encoding_size,
                             batch_first = True)
        self.variableLSTMs = nn.ModuleDict()                                    #Create the dictionary for the variable decoder LSTM networks
        self.variableLinears = nn.ModuleDict()                                  #Create the dictionary for the variable linear layer for the reconstruction from LSTM outputs, as done in Malhotra et al. (2016)
        self.teacher_forcing = teacher_forcing                                  #Optionally, use teacher forcing
        for shifting_type in self.combinations:                                 
            for transformer_ID in self.combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type, transformer_ID, self.combinations)
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
                _ , (h_t, c_t) = self.fixed(x_i)                                #Run the data through the encoder and retrieve the encoding
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

class GRU_MAE(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations, device, teacher_forcing:bool = True):
        super(GRU_MAE, self).__init__()
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.device = device
        self.fixed = nn.GRU(input_size = 1,                                     #Create the fixed encoder network
                            hidden_size = self.encoding_size,
                            batch_first = True)
        self.variableGRUs = nn.ModuleDict()                                     #Create the dictionary for the variable decoder GRU networks
        self.variableLinears = nn.ModuleDict()                                  #Create the dictionary for the variable linear layer for the reconstruction from GRU outputs, as done in Malhotra et al. (2016)
        self.teacher_forcing = teacher_forcing                                  #Optionally, use teacher forcing
        for shifting_type in combinations:                                      
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
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
                _, h_t = self.fixed(x_i)                                        #Run the data through the encoder and retrieve the encoding
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


def create_dataset(combinations, cols, split:bool = False, train:bool = True, ratio:float = 0.8, path:str = "C:\LocalData\Data\Raw_data"):
    my_dataset = TorqueDataset(cols, combinations, split, train, ratio, path)                           
    return my_dataset

def create_dataloader(dataset:TorqueDataset, batch_size:int,  max_batches_per_ID:int = 1_000_000, complete_batches_only:bool = False):
    my_dataloader = DataLoader(dataset= dataset, batch_sampler=Batch_IDs_together(dataset, batch_size, max_batches_per_ID, complete_batches_only))
    return my_dataloader

def create_mixed_batch_dataloader(dataset:TorqueDataset, batch_size_per_ID:int):
    my_dataloader = DataLoader(dataset = dataset, batch_sampler=Batch_mixed_IDs(dataset, batch_size_per_ID))
    return my_dataloader

def train_model(model, num_epochs, dataloader, loss_function, optimizer, device,
                sequence_length, learning_rate, save_model:bool = False,
                save_model_path:str = "",
                convergence_plot:bool = True, linewidth = 0.2,
                return_training_loss:bool = False, lr_schedule:bool = False,
                schedule_milestones:list = [60,90]):
    model.train()
    loss_function = loss_function()
    optimizer = optimizer(model.parameters(), learning_rate)
    if lr_schedule:
        scheduler = MultiStepLR(optimizer, milestones=schedule_milestones, gamma=0.1)
    losses = []                                                                 #Keep track of the loss of each batch
    for epoch in tqdm(range(num_epochs),leave=None):
        last_losses = []
        for data in dataloader:
            data= data.to(device=device)                                        #Bring the data onto the device
            reconstruction = model(data)                                        #Execute the model
            loss = loss_function(reconstruction, data[:,:sequence_length])      #Compute the loss
            optimizer.zero_grad()                                               #Set the gradients to 0 for each batch
            loss.backward()                                                     #Compute gradients using backpropagation
            optimizer.step()                                                    #Execute parameter updates
            losses.append(loss.data)
            last_losses.append(loss.data)
        if lr_schedule:
            scheduler.step()
    if save_model:
        if save_model_path == "":
            raise ValueError("Enter a path to the location where the model must be saved.")
        else:
            torch.save(model, save_model_path)        
    if convergence_plot:                                                        #Optionally plot the convergence over the iterations
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.ylim(top = 20)
        plt.plot(torch.Tensor(losses).cpu(), linewidth = linewidth, color = "black")
        plt.show()
    if return_training_loss:
        return torch.mean(torch.Tensor(losses).cpu())

def create_Vanilla_MAE(encoding_size, sequence_length, combinations,
                        n_hidden_layers, device, hidden_size:int = -1, n_variable_layers:int = -1):
    model = Vanilla_MAE(encoding_size, sequence_length, combinations, 
                        n_hidden_layers, hidden_size, n_variable_layers).to(device)
    return model

def create_TCN_MAE(sequence_length, combinations, n_channels, n_channels_reduced,
                    n_encoding_channels, n_variable_layers, device, kernel_size,
                    pool_rate, num_layers):
    model = TCN_MAE(sequence_length, combinations, n_channels, n_channels_reduced, 
                    n_encoding_channels, n_variable_layers, device, kernel_size, 
                    pool_rate, num_layers).to(device)
    return model

def create_LSTM_MAE(encoding_size, sequence_length, combinations, device, teacher_forcing:bool = True):
    model = LSTM_MAE(encoding_size, sequence_length, combinations, device, teacher_forcing).to(device)
    return model

def create_GRU_MAE(encoding_size, sequence_length, combinations, device, teacher_forcing:bool = True):
    model = GRU_MAE(encoding_size, sequence_length, combinations, device, teacher_forcing).to(device)
    return model

def evaluate_on_dataset(model, data, loss_function, device, batch_size:int = 256, show_roc_curve:bool = True, shared_threshold:bool = True):
    model.eval()                                                                #Set the model into evaluation mode
    loss_function = loss_function(reduction = 'none')                           #Make the loss function compute individual losses for observations instead of one aggregate loss for the entire batch
    data = torch.Tensor(data.values).to(device)                                 #Bring the data to the device in tensor format
    if shared_threshold:
        n_chunks = round(data.shape[0]/batch_size)                              #Compute the number of chunks required to match the batch size as closely as possible
        all_reconstruction_errors = torch.empty(0).to(device)                   #Create a tensor to save all reconstruction errors
        for chunk in torch.tensor_split(data, n_chunks):                        #Iterate over distinct chunks of the data
            output = model(chunk[:,:-1]).data                                   #Run the data through the model
            reconstruction_errors = torch.mean(loss_function(chunk[:,:-2], output), dim = 1) #Compute the reconstruction errors
            all_reconstruction_errors = torch.cat((all_reconstruction_errors, reconstruction_errors)) #Save the reconstruction errors in the tensor
        fpr,tpr,thresholds = metrics.roc_curve(data[:,-1].cpu(), all_reconstruction_errors.cpu()) #Comput the ROC data (has to be on cpu to use numpy)
        auc = metrics.auc(fpr,tpr)                                              #Compute the AUC
        if show_roc_curve:                                                      #Plot the result
            plt.plot(fpr, tpr, color = 'black', label = f"ROC curve (area = {auc}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc = "lower right")
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.show()
        return auc, (fpr, tpr)
    else:
        ids = torch.unique(data[:,-2],sorted=False)                             #Retrieve the ids present in the data
        aucs = {}
        fprs = {}
        tprs = {}
        for id in ids:                                                          #Iterate over these ids and execute the same steps as above
            reconstruction_errors=[]
            data_id = data[data[:,-2] == id]
            n_chunks_id = round(data_id.shape[0]/batch_size)
            all_reconstruction_errors = torch.empty(0).to(device)
            for chunk in torch.tensor_split(data_id, n_chunks_id):
                output = model(chunk[:,:-1]).data
                reconstruction_errors = torch.mean(loss_function(chunk[:,:-2], output), dim=1)
                all_reconstruction_errors = torch.cat((all_reconstruction_errors, reconstruction_errors))
            fpr,tpr,thresholds = metrics.roc_curve(data_id[:,-1].cpu(), all_reconstruction_errors.cpu())
            auc = metrics.auc(fpr,tpr)
            aucs[id] = auc
            fprs[id] = fpr
            tprs[id] = tpr
            if show_roc_curve:
                plt.plot(fpr,tpr, label = f"Task {int(id)} (area = {round(auc,4)}")
        avg_auc = sum(aucs.values())/len(aucs.values())
        if show_roc_curve:
            plt.plot([0,1], [0,0], color = 'white', label = f"Average area = {round(avg_auc,5)}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc = "lower right")
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
        return avg_auc, (fprs, tprs)
