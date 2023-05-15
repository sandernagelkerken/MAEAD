import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from functions.combinations import combination_to_id, id_to_combination
from functions.dataloader import load_torque

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

class SimpleMultitaskAutoencoder(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations):
        super(SimpleMultitaskAutoencoder, self).__init__()                      #Initialize the network
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length                                  
        self.combinations = combinations
        self.l1 = nn.Linear(sequence_length, encoding_size)                     #Define the first layer
        self.l2 = nn.ModuleDict()                                               #Define a dictionary of modules (a dictionary of layers)
        for shifting_type in combinations:                                      #Iterate over all combinations shifting types and transformer IDs
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) #Get the combination ID of the combination
                self.l2[str(combination_ID)] = nn.Linear(encoding_size, sequence_length)   #Create a unique layer for the specific combination and store this in the dictionary of modules

    def forward(self, x):
        y = []
        ids = torch.unique(x[:,self.sequence_length],sorted=False)
        for i in ids:
            x_i = x[x[:,self.sequence_length] == i]
            if x_i.shape[0] != 0:
                combination_ID = str(int(x_i[0,self.sequence_length].item()))             #Read the combination ID of first element of the batch
                x_i = x_i[:,:self.sequence_length]                                          #Use the columns without the combination ID for the computations
                x_i = F.relu(self.l1(x_i))                                                  #First layer, always the same
                x_i = self.l2[combination_ID](x_i)                                          #Second layer, dependent on the combination ID
                y.append(x_i)
        x = torch.concat(y,0)
        return x

class ConvolutionalMultitaskAutoencoder(nn.Module):
    def __init__(self, sequence_length, combinations, n_channels:int, n_variable_layers:int, maxpool_rate:int = 2):
        super(ConvolutionalMultitaskAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.n_channels = n_channels
        self.maxpool_rate = maxpool_rate
        self.n_variable_layers = n_variable_layers
        self.encoder = nn.Sequential(nn.Conv1d(1,self.n_channels,5),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(self.n_channels, self.n_channels, 5, dilation=2),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(self.n_channels, self.n_channels, 5, dilation=4),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(self.n_channels, self.n_channels, 5, dilation=8),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(self.n_channels, self.n_channels, 5, dilation=16),
                                     nn.LeakyReLU(),
                                     nn.Conv1d(self.n_channels, self.n_channels, 5, dilation=32),
                                     nn.LeakyReLU(),
                                     nn.MaxPool1d(maxpool_rate, return_indices=True)
                                     )
        self.unpool = nn.MaxUnpool1d(maxpool_rate)
        self.decoder_fixed = nn.Sequential()
        for i in range(6 - self.n_variable_layers):
            self.decoder_fixed.add_module(f"decoding layer {i}", nn.ConvTranspose1d(self.n_channels, self.n_channels, 5, dilation=2**(5-i)))
            self.decoder_fixed.add_module("ReLU", nn.LeakyReLU())
        self.decoder_variable = nn.ModuleDict()
        for shifting_type in combinations:                                      
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                self.decoder_variable[str(combination_ID)] = nn.Sequential()
                for i in range(self.n_variable_layers):
                    j = (6 - self.n_variable_layers) + i
                    self.decoder_variable[str(combination_ID)].add_module(f"decoding layer {j}", nn.ConvTranspose1d(self.n_channels, self.n_channels, 5, dilation=2**(5-j)))
                    self.decoder_variable[str(combination_ID)].add_module("ReLU", nn.LeakyReLU())
                self.decoder_variable[str(combination_ID)].add_module("final decoding layer", nn.Conv1d(self.n_channels, 1, 1))
        
    def forward(self, x):
        y = []
        ids = torch.unique(x[:,self.sequence_length],sorted=False)
        for i in ids:
            x_i = x[x[:,self.sequence_length] == i]
            if x_i.shape[0] != 0:
                combination_ID = str(int(x_i[0,self.sequence_length].item()))    #Read the combination ID of first element of the batch
                x_i = x_i[:,:self.sequence_length] 
                x_i = x_i.unsqueeze(1)
                encoding, indices = self.encoder(x_i)
                x_i = self.unpool(encoding,indices)
                x_i = self.decoder_fixed(x_i)
                x_i = self.decoder_variable[combination_ID](x_i)
                x_i = x_i.squeeze(1)                                        #Second layer, dependent on the combination ID
                y.append(x_i)
        x = torch.concat(y,0)
        return x

# class GRUMulitiTaskAutoencoder(nn.Module):
#     def __init__(self, encoding_size, sequence_length, combinations):
#         super(GRUMulitiTaskAutoencoder, self).__init__()
#         self.encoding_size = encoding_size
#         self.sequence_length = sequence_length
#         self.combinations = combinations
#         self.fixed = nn.GRU(input_size = 1, 
#                             hidden_size = self.encoding_size,
#                             batch_first = True)
#         self.variableGRUs = nn.ModuleDict()
#         # self.variableLinears = nn.ModuleDict()
#         for shifting_type in combinations:                                      
#             for transformer_ID in combinations[shifting_type]:
#                 combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
#                 self.variableGRUs[str(combination_ID)] = nn.GRU(input_size = self.encoding_size, hidden_size = self.encoding_size)
#                 # self.variableLinears[str(combination_ID)] = nn.Linear(self.encoding_size, 1)
            
#     def forward(self, x):
#         y=[]
#         ids = torch.unique(x[:,self.sequence_length],sorted=False)
#         for i in ids:
#             x_i = x[x[:,self.sequence_length] == i]
#             if x_i.shape[0] != 0:
#                 combination_ID = str(int(x_i[:,self.sequence_length].data.numpy()[0]))    #Read the combination ID of first element of the batch
#                 x_i = x_i[:,:self.sequence_length]                                          #Use the columns without the combination ID for the computations
#                 x_i = x_i.unsqueeze(2)
#                 _, encoding = self.fixed(x_i)
#                 encoding_repeated = encoding.expand(self.sequence_length, -1, -1)
#                 output, h_t = self.variableGRUs[combination_ID](encoding_repeated)
#                 output = output.sum(2)
#                 output = torch.transpose(output, 0, 1)
#                 y.append(output)
#         x = torch.concat(y,0)
#         return x

class GRUMulitiTaskAutoencoder(nn.Module):
    def __init__(self, encoding_size, sequence_length, combinations):
        super(GRUMulitiTaskAutoencoder, self).__init__()
        self.encoding_size = encoding_size
        self.sequence_length = sequence_length
        self.combinations = combinations
        self.fixed = nn.GRU(input_size = 1, 
                            hidden_size = self.encoding_size,
                            batch_first = True)
        self.variableGRUs = nn.ModuleDict()
        self.variableLinears = nn.ModuleDict()
        for shifting_type in combinations:                                      
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) 
                self.variableGRUs[str(combination_ID)] = nn.GRU(input_size = 1, hidden_size = self.encoding_size)
                self.variableLinears[str(combination_ID)] = nn.Linear(self.encoding_size, 1)
            
    def forward(self, x):
        y=[]
        ids = torch.unique(x[:,self.sequence_length],sorted=False)
        for i in ids:
            x_i = x[x[:,self.sequence_length] == i]
            if x_i.shape[0] != 0:
                batch_size = x_i.shape[0]
                combination_ID = str(int(x_i[0,self.sequence_length].item()))
                x_i = x_i[:,:self.sequence_length]                                          
                x_i = x_i.unsqueeze(2)
                _, h_t = self.fixed(x_i)
                outputs = torch.zeros(batch_size, self.sequence_length)
                output = torch.zeros(1,batch_size)
                for step in range(self.sequence_length):
                    output = output.unsqueeze(2)
                    output, h_t = self.variableGRUs[combination_ID](output, h_t)
                    output = self.variableLinears[combination_ID](output)
                    outputs[:,step] = output        
                y.append(outputs)
        x = torch.concat(y,0)
        return x

class TwoTwoOneMultitaskAutoencoder(nn.Module):                                 #2 input layers, 2 output layers, 1 variable layer
    def __init__(self, encoding_size, inter_layer_size, sequence_length, combinations):
        super(TwoTwoOneMultitaskAutoencoder, self).__init__()                      #Initialize the network
        self.encoding_size = encoding_size
        self.inter_layer_size = inter_layer_size
        self.sequence_length = sequence_length                                  
        self.combinations = combinations
        self.fixed = nn.Sequential(nn.Linear(sequence_length, inter_layer_size),           #Define the shared encoding network
                                     nn.LeakyReLU(),
                                     nn.Linear(inter_layer_size,encoding_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(encoding_size, inter_layer_size),
                                     nn.LeakyReLU())
        self.variable = nn.ModuleDict()                                         #Define a dictionary of modules for the variable part
        for shifting_type in combinations:                                      #Iterate over all combinations shifting types and transformer IDs
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) #Get the combination ID of the combination
                self.variable[str(combination_ID)] = nn.Linear(inter_layer_size, sequence_length) #Create a unique layer for the specific combination and store this in the dictionary of modules

    def forward(self, x):
        y = []
        ids = torch.unique(x[:,self.sequence_length],sorted=False)
        for i in ids:
            x_i = x[x[:,self.sequence_length] == i]
            if x_i.shape[0] != 0:
                combination_ID = str(int(x_i[0,self.sequence_length].item()))    #Read the combination ID of first element of the batch
                x_i = x_i[:,:self.sequence_length]                                          #Use the columns without the combination ID for the computations
                x_i = self.fixed(x_i)                                                       #First layer, shared between tasks                                              
                x_i = self.variable[combination_ID](x_i)                                          #Second layer, dependent on the combination ID
                y.append(x_i)
        x = torch.concat(y,0)
        return x

class TwoTwoTwoMultitaskAutoencoder(nn.Module):                                   #2 input layers, 2 output layers, last 2 layers are variable
    def __init__(self, encoding_size, inter_layer_size, sequence_length, combinations):
        super(TwoTwoTwoMultitaskAutoencoder, self).__init__()                      #Initialize the network
        self.encoding_size = encoding_size
        self.inter_layer_size = inter_layer_size
        self.sequence_length = sequence_length                                  
        self.combinations = combinations
        self.encoder = nn.Sequential(nn.Linear(sequence_length, inter_layer_size),           #Define the shared encoding network
                                     nn.LeakyReLU(),
                                     nn.Linear(inter_layer_size,encoding_size),
                                     nn.LeakyReLU())
        self.decoder = nn.ModuleDict()                                          #Define a dictionary of modules for the decoder
        for shifting_type in combinations:                                      #Iterate over all combinations shifting types and transformer IDs
            for transformer_ID in combinations[shifting_type]:
                combination_ID = combination_to_id(shifting_type,transformer_ID, combinations) #Get the combination ID of the combination
                self.decoder[str(combination_ID)] = nn.Sequential(nn.Linear(encoding_size, inter_layer_size),
                                                                  nn.LeakyReLU(),
                                                                  nn.Linear(inter_layer_size, sequence_length))   #Create a unique layer for the specific combination and store this in the dictionary of modules

    def forward(self, x):
        y = []
        ids = torch.unique(x[:,self.sequence_length],sorted=False)
        for i in ids:
            x_i = x[x[:,self.sequence_length] == i]
            if x_i.shape[0] != 0:
                combination_ID = str(int(x_i[0,self.sequence_length].item()))    #Read the combination ID of first element of the batch
                x_i = x_i[:,:self.sequence_length]                                          #Use the columns without the combination ID for the computations
                x_i = self.encoder(x_i)                                                     #First layer, shared between tasks                                              
                x_i = self.decoder[combination_ID](x_i)                                         #Second layer, dependent on the combination ID
                y.append(x_i)
        x = torch.concat(y,0)
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

def create_multitaskAE_model(encoding_size, sequence_length, combinations, device):
    my_model = SimpleMultitaskAutoencoder(encoding_size, sequence_length, combinations).to(device)
    return my_model

def create_TwoTwoOne_multitaskAE_model(encoding_size, inter_size, sequence_length, combinations, device):
    my_model = TwoTwoOneMultitaskAutoencoder(encoding_size, inter_size, sequence_length, combinations).to(device)
    return my_model

def create_TwoTwoTwo_multitaskAE_model(encoding_size, inter_size, sequence_length, combinations, device):
    my_model = TwoTwoTwoMultitaskAutoencoder(encoding_size, inter_size, sequence_length, combinations).to(device)
    return my_model

def create_GRU_multitaskAE_model(encoding_size, sequence_length, combinations, device):
    my_model = GRUMulitiTaskAutoencoder(encoding_size, sequence_length, combinations).to(device)
    return my_model

def create_TCN_multitaskAE_model(n_channels, n_variable_layers, maxpool_rate, sequence_length, combinations, device):
    my_model = ConvolutionalMultitaskAutoencoder(sequence_length, combinations, n_channels, n_variable_layers, maxpool_rate).to(device)
    return my_model

def train_model(model:SimpleMultitaskAutoencoder, num_epochs, dataloader, loss_function, optimizer, device,
                sequence_length, learning_rate, save_model:bool = False,
                save_model_path:str = "",
                convergence_plot:bool = True, linewidth = 0.2,
                return_training_loss:bool = False):
    loss_function = loss_function()
    optimizer = optimizer(model.parameters(), learning_rate)
    model.train()
    losses = []                                                                 #Keep track of the loss of each batch
    for epoch in tqdm(range(num_epochs),leave=None):
        last_losses = []
        for data in dataloader:
            data= data.to(device=device) 
            reconstruction = model(data)                                        #Execute the model
            loss = loss_function(reconstruction, data[:,:sequence_length])      #Compute the loss
            optimizer.zero_grad()                                               #Set the gradients to 0 for each batch
            loss.backward()                                                     #Compute gradients using backpropagation
            optimizer.step()                                                    #Execute parameter updates
            losses.append(loss.data)
            last_losses.append(loss.data)
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
        return np.mean(last_losses)

def plot_reconstructions(model: SimpleMultitaskAutoencoder, dataloader, combinations,                       
                        device, sequence_length, num_plots:int = 10):           #Plotting function to inspect results intuitively
    model.eval()
    counter = 0
    for data in dataloader:
        data= data.to(device=device)
        combination_ID = int(data.data.numpy()[0,sequence_length])
        reconstruction = model(data)
        plt.plot(reconstruction.data[0], color = "red", label = 'Reconstruction')
        plt.plot(data.data[0,:sequence_length], color = "black", label = 'Original')
        plt.legend()
        plt.xlabel("Operational progress in terms of rotation")
        plt.ylabel("Torque measurement in Nm")
        plt.xticks(ticks = np.arange(0,301,30), labels= [f"{x}%" for x in np.arange(0,101,10)])
        shifting_type, transformer_ID = id_to_combination(combination_ID, combinations)
        plt.title(f"Random sample from a {shifting_type} operation of transformer {transformer_ID}")
        plt.show()
        counter+=1
        if counter>num_plots-1:
            break