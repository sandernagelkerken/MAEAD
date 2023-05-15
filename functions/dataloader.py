import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_envelope(shifting_type:str, transformer_id:int,
                    path:str = "C:/LocalData/Data/Raw_data"):
    df_envelope = pd.read_parquet(f"{path}/ENVELOPE/{shifting_type}/ENVELOPE_{shifting_type}_{str(transformer_id)}.parquet")
    return df_envelope
    
def load_master_data(shifting_type:str, transformer_id:int, 
                    path:str = "C:/LocalData/Data/Raw_data"):
    df_master_data = df_torque = pd.read_parquet(f"{path}/MASTER_DATA/{shifting_type}/MASTER_DATA_{shifting_type}_{str(transformer_id)}.parquet")
    return df_master_data

def load_threshold(shifting_type:str, transformer_id:int, 
                    path:str = "C:/LocalData/Data/Raw_data"):
    df_threshold = pd.read_parquet(f"{path}/THRESHOLD/{shifting_type}/THRESHOLD_{shifting_type}_{str(transformer_id)}.parquet")        
    return df_threshold

def load_torque(shifting_type:str, transformer_id:int, 
                    path:str = "C:/LocalData/Data/Raw_data",
                    fill_NA_values:bool = False):
    df_torque = pd.read_parquet(f"{path}/TORQUE/{shifting_type}/TORQUE_{shifting_type}_{str(transformer_id)}.parquet")
    if fill_NA_values:
        df_torque = df_torque.fillna(0)
    return df_torque

def plot_torque(shifting_type:str, transformer_id:int, 
                path:str = "C:/LocalData/Data/Raw_data",
                fill_NA_values:bool = False):
    df = load_torque(shifting_type=shifting_type, transformer_id=transformer_id,
                        path=path, fill_NA_values=fill_NA_values)
    ax = plt.subplot()
    ax.set_title(f"{shifting_type}: {transformer_id}")
    ax.set_xticks(ticks = np.arange(0,301,25), labels= np.arange(0,301,25))
    ax.set_ylabel("Torque in Nm")
    df.T.plot(color = 'black', linewidth = 0.05, legend = False, ax = ax)
    plt.show()