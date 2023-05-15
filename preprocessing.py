import pandas as pd
from tqdm import tqdm
from functions.dataloader import load_envelope, load_torque
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def remove_obvious_outliers(dataframe:pd.DataFrame, num_stds:int = 3, show_size_reduction:bool = True,
                            remove_flatlines:bool = True):
    if remove_flatlines:
        cleandf = dataframe.loc[(dataframe!=0).any(axis=1)]
        if show_size_reduction:
            print(f"=======FLATLINES=======")
            print(f"Original size: {len(dataframe)}")
            print(f"New size:      {len(cleandf)}")
            print(f"Size reduction:{round(100-(100*len(cleandf)/len(dataframe)),1)}%")
    else:
        cleandf = dataframe
    means = cleandf.mean(0)
    stds = cleandf.std(0)
    for i in range(0,295):
        cleandf = cleandf[cleandf[str(i)].between(means[i]-num_stds*stds[i], means[i]+num_stds*stds[i])]
    if show_size_reduction:
        print(f"=======STD OUTLIERS=======")
        print(f"Original size: {len(dataframe)}")
        print(f"New size:      {len(cleandf)}")
        print(f"Size reduction:{round(100-(100*len(cleandf)/len(dataframe)),1)}%")
    return cleandf

def compute_derivatives(df, 
                        rename_columns:bool =True):
    df_derivative = df.fillna(0).diff(axis=1)
    if rename_columns:
        def rename_derivatives(column_name):
            return f"{column_name}^1"
        df_derivative = df_derivative.rename(columns = rename_derivatives)
    return df_derivative.fillna(0)

def find_most_normal_curves(dataframe:pd.DataFrame, 
                            n_curves:int = 1, 
                            interval:float = 0.1):
    sd = 0
    count = 0
    most_normal_data = remove_obvious_outliers(dataframe, sd, False)
    while len(most_normal_data) < n_curves:
        sd+= interval
        most_normal_data = remove_obvious_outliers(dataframe, sd, False)
    if len(most_normal_data) > n_curves:
        sd -= interval
        interval = interval/10
        count += 1
        most_normal_data = remove_obvious_outliers(dataframe, sd, False)
        while len(most_normal_data) == 0:
            sd += interval
            most_normal_data = remove_obvious_outliers(dataframe, sd, False)
    most_normal_data = most_normal_data.iloc[:n_curves]
    print(f"The {n_curves} most normal curve(s) stay(s) within {round(sd,3)} standard deviations of the mean curve.")
    return most_normal_data

def mean_curve(dataframe:pd.DataFrame):
    means = dataframe.mean(0)
    return list(means)

def sd_curve(dataframe:pd.DataFrame):
    stds = dataframe.std(0)
    return list(stds)

def automated_preprocessing(shifting_type:str,
                           transformer_ID,
                           filter_first_rows:bool = False,
                           n_first_rows:int = 200,
                           filter_envelope:bool = False,
                           filter_flatline:bool = True,
                           filter_std:bool = True,
                           n_std:float = 5,
                           source_path:str = "C:\LocalData\Data\Manually_preprocessed_data",
                           save_processed:bool = False,
                           target_path: str = "C:\LocalData\Data\Completely_preprocessed_data"):
    #Load the original datasets
    df_torque = load_torque(shifting_type, transformer_ID, source_path, fill_NA_values=True)
    #Step 1: Remove the first number of rows
    if filter_first_rows:
        df_torque = df_torque.iloc[n_first_rows:]
    #Step 2: remove envelope exceeds
    if filter_envelope:
        df_envelope = load_envelope(shifting_type, transformer_ID, "C:\LocalData\Data\Raw_data")
        df_torque = df_torque[df_torque.le(df_envelope, axis=0)].dropna()
    #Step 3: remove flatlines
    if filter_flatline:
        df_torque = df_torque.loc[(df_torque!=0).any(axis=1)]
    #Step 4: remove standard deviation ouliers of the MSE
    if filter_std:
        median = df_torque.median(0)
        mses = df_torque.apply(lambda row : mean_squared_error(row, median), axis = 1, raw = True)
        median_mse, std_mse = np.median(mses), np.std(mses)
        for i, row in df_torque.iterrows():
            mse = mean_squared_error(row,median)
            if mse > median_mse+n_std*std_mse:
                df_torque.drop(i, inplace=True)
    #Use processed dataset
    if save_processed:
        df_torque.to_parquet(f"{target_path}\TORQUE\{shifting_type}\TORQUE_{shifting_type}_{str(transformer_ID)}.parquet")
    return df_torque

def plot_preprocessing_result(shifting_type, 
                              transformer_ID, 
                              colored_removes:bool = False, 
                              original_path:str = "C:\LocalData\Data\Raw_data", 
                              processed_path:str = "C:\LocalData\Data\Completely_preprocessed_data"):
    df_original = load_torque(shifting_type, transformer_ID, original_path)
    max_value = df_original.max()
    try:
        df_processed = load_torque(shifting_type, transformer_ID, processed_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The data for {shifting_type} {transformer_ID} has not yet been processed")
    df_removed = pd.concat([df_original, df_processed, df_processed]).drop_duplicates(keep=False)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize = (20,5))
    ax1.set_title("Original data")
    ax1.set_ylabel("Torque measurement in Nm")
    if max_value > 1000:
        ax1.set_ylim((0,200))
    ax2.set_title("Preprocessed data")
    ax3.set_title("Removed data")
    fig.suptitle(f"Preprocessing result for {shifting_type} {transformer_ID}")
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(ticks = np.arange(0,301,30), labels= [f"{x}%" for x in np.arange(0,101,10)])
        ax.set_xlabel("Operational progress in terms of rotation")
    if len(df_processed) > 0:
        df_processed.T.plot(legend = False, color = 'black', linewidth = 0.01, ax = ax1)
        df_processed.T.plot(legend = False, color = 'black', linewidth = 0.01, ax = ax2)
    if len(df_removed) > 0:
        if colored_removes:
            df_removed.T.plot(legend = False, color = 'red', linewidth = 0.01, ax = ax1)
            df_removed.T.plot(legend = False, color = 'black', linewidth = 0.01, ax = ax3)
        else:
            df_removed.T.plot(legend = False, color = 'black', linewidth = 0.01, ax = ax1)
            df_removed.T.plot(legend = False, color = 'black', linewidth = 0.01, ax = ax3)
    plt.show()
    pass

def remove_curves_in_area(shifting_type:str, 
                          transformer_ID, 
                          threshold:float, 
                          apply_at_range:tuple = (),
                          reverse:bool = False, 
                          plot_result:bool = False,
                          origin_path:str = "C:\LocalData\Data\Manually_preprocessed_data",
                          write_to_target:bool = False,
                          target_path:str = "C:\LocalData\Data\Manually_preprocessed_data",):
    df = load_torque(shifting_type, transformer_ID, origin_path, True)
    if reverse:
        thresholds = [-0.1] * df.shape[1]
    else:
        thresholds = [1_000_000] * df.shape[1]
    if apply_at_range == ():
        thresholds = [threshold] *df.shape[1]
    else:
        for i in range(apply_at_range[0], apply_at_range[1]):
            thresholds[i] = threshold
    if reverse:
        new_df = df[df.ge(thresholds)].dropna()
    else:
        new_df = df[df.le(thresholds)].dropna()
    if plot_result:
        max_value = df.max().max()
        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize = (15,5))
        ax1.set_xticks(ticks = np.arange(0,301,25), labels= np.arange(0,301,25))
        ax2.set_xticks(ticks = np.arange(0,301,25), labels= np.arange(0,301,25))
        df.T.plot(ax = ax1, color = 'black', linewidth = 0.05, legend = False, ylim = (0,max_value), ylabel = "Torque measurement in Nm")
        threshold, = ax1.plot(thresholds, color = 'red', label = "Threshold")
        ax1.legend(handles = [threshold])
        new_df.T.plot(ax = ax2, color = 'black', linewidth = 0.05, legend = False)
        plt.show()
    if write_to_target:
        new_df.to_parquet(f"{target_path}\TORQUE\{shifting_type}\TORQUE_{shifting_type}_{str(transformer_ID)}.parquet")
    return new_df

def remove_curves_with_many_na(shifting_type:str,
                               transformer_ID,
                               max_na_per_curve:int = 25,
                               origin_path:str = "C:/LocalData/Data/Raw_data",
                               write_to_target:bool = False,
                               target_path:str = "C:/LocalData/Data/NA_removed_data"):
    df = load_torque(shifting_type, transformer_ID, origin_path)
    new_df = df[df.isna().sum(axis= 1) <= max_na_per_curve]
    if write_to_target:
        new_df.to_parquet(f"{target_path}\TORQUE\{shifting_type}\TORQUE_{shifting_type}_{str(transformer_ID)}.parquet")
    return new_df

def remove_curves_with_large_values(shifting_type:str,
                                    transformer_ID,
                                    max_value:int = 1000,
                                    origin_path:str = "C:/LocalData/Data/NA_removed_data",
                                    write_to_target:bool = False,
                                    target_path:str = "C:/LocalData/Data/Manually_preprocessed_data"):
    df = load_torque(shifting_type, transformer_ID, origin_path, True)
    new_df = df[df.le(max_value)].dropna()
    if write_to_target:
        new_df.to_parquet(f"{target_path}\TORQUE\{shifting_type}\TORQUE_{shifting_type}_{str(transformer_ID)}.parquet")
    return new_df
