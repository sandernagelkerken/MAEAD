from os import listdir
from functions.dataloader import load_torque
import matplotlib.pyplot as plt
import pandas as pd

def all_combinations(path = "C:/LocalData/Data/Raw_data", 
                     normal_lengths_only:bool = False,
                     min_dataset_size:int = 0,
                     max_dataset_size:int = 1_000_000_000,
                     max_na_per_curve:int = 1000):
    path_torque = path+"/TORQUE"
    shifting_types = listdir(path_torque)
    combinations = {}
    for shifting_type in shifting_types:
        combinations[shifting_type] = []
        typepath = path_torque+"/"+str(shifting_type)
        files = listdir(typepath)
        for file in files:
            transformer_ID = file.split("_")[2].split(".")[0]
            if normal_lengths_only or (min_dataset_size != 0) or (max_dataset_size != 1_000_000_000):
                df = load_torque(shifting_type, transformer_ID, path)
                df = df[df.isna().sum(axis= 1) < max_na_per_curve]
                df_shape = df.shape
                if normal_lengths_only:
                    if df_shape[1] in range(299,305):
                        if min_dataset_size <= df_shape[0]:
                            if max_dataset_size >= df_shape[0]:
                                combinations[shifting_type].append(transformer_ID)
                else:
                    if min_dataset_size <= df_shape[0]:
                        if max_dataset_size >= df_shape[0]:
                            combinations[shifting_type].append(transformer_ID)
            else:
                combinations[shifting_type].append(transformer_ID)
    for shifting_type in combinations:
        combinations[shifting_type] = sorted([int(i) for i in combinations[shifting_type]])
    return combinations

def get_all_combinations_with_IDs(normal_lengths_only:bool = False, min_dataset_size:int = 0, combinations:dict = None):
    if combinations == None:
        combinations = all_combinations(normal_lengths_only=normal_lengths_only, min_dataset_size=min_dataset_size)
    combinations_IDs = {}
    i=0
    for shifting_type in combinations:
        combinations_IDs[shifting_type] = {}
        for transformer_ID in combinations[shifting_type]:
            combinations_IDs[shifting_type][transformer_ID] = i
            i += 1
    return combinations_IDs

def combination_to_id(shifting_type:str, transformer_ID, combinations_with_IDs):
    return combinations_with_IDs[shifting_type][transformer_ID]

def id_to_combination(id, combinations_with_IDs):
    for shifting_type in combinations_with_IDs:
        for transformer_ID in combinations_with_IDs[shifting_type]:
            if combinations_with_IDs[shifting_type][transformer_ID] == int(id):
                return shifting_type, transformer_ID

def plot_dataset_size_distribution(combinations, max_dataset_size:int = 1_000_000_000, 
                                    path:str = "C:\LocalData\Data\Completely_preprocessed_data"):
    all_combinations = combinations
    combinations_sizes = []
    for shifting_type in all_combinations:
        for transformer_ID in all_combinations[shifting_type]:
            combination_df_torque = load_torque(shifting_type, transformer_ID, path)
            combination_size = len(combination_df_torque)
            if combination_size <= max_dataset_size:
                combinations_sizes.append(combination_size)
    plt.hist(combinations_sizes, color="black", bins = 100)
    plt.xlabel("Number of sequences for a task")
    plt.ylabel("Frequency")
    plt.show()
    return combinations_sizes

def plot_sequence_length_distribution(combinations):
    all_combinations = combinations
    combinations_lengths = []
    for shifting_type in all_combinations:
        for transformer_ID in all_combinations[shifting_type]:
            combination_df_torque = load_torque(shifting_type, transformer_ID)
            combination_length = combination_df_torque.shape[1]
            combinations_lengths.append(combination_length)
    plt.hist(combinations_lengths, color="black", bins = 20)
    plt.xlabel("Length of the sequences for a task")
    plt.ylabel("Frequency")
    plt.show()
    return combinations_lengths

def get_n_combinations(combinations:dict):
    count = 0
    for shifting_type in combinations:
        for transformer_ID in combinations[shifting_type]:
            count += 1
    return count

def get_closest_combinations(target_size:int, n_combinations:int, split_sets:bool = False, path:str = "C:\LocalData\Data\Completely_preprocessed_data",
                             return_df:bool = False):
    combinations = all_combinations(path, True)
    combinations_df = pd.DataFrame(columns=["Shifting type", "Transformer ID", "Size"])
    for shifting_type in combinations:
        for transformer_ID in combinations[shifting_type]:
            size = len(load_torque(shifting_type, transformer_ID, path))
            row = [shifting_type, transformer_ID, size]
            combinations_df.loc[len(combinations_df)] = row
    plusminus, count = 1, 1
    sub_df = combinations_df.head(0)
    while len(sub_df) < n_combinations:
        target_df = combinations_df[combinations_df['Size'] == target_size]
        for i in range(len(target_df)):
            sub_df.loc[len(sub_df)] = target_df.iloc[i]
        target_size += plusminus * count
        plusminus = plusminus * -1
        count += 1
    if return_df:
        return sub_df
    if split_sets:
        val_combinations = all_combinations(path, True, -1,-1)
        test_combinations = all_combinations(path, True, -1,-1)
        for i in range(len(sub_df)):
            row = sub_df.iloc[i]
            if i%2 == 0:
                val_combinations[row[0]].append(row[1])
            else:
                test_combinations[row[0]].append(row[1])
        val_combinations_with_ID = get_all_combinations_with_IDs(combinations=val_combinations)
        test_combinations_with_ID = get_all_combinations_with_IDs(combinations=test_combinations)
        return val_combinations_with_ID, test_combinations_with_ID
    else:
        target_combinations = all_combinations(path, True, -1,-1)
        for i in range(len(sub_df)):
            row = sub_df.iloc[i]
            target_combinations[row[0]].append(row[1])
        target_combinations_with_ID = get_all_combinations_with_IDs(combinations=target_combinations)
        return target_combinations_with_ID