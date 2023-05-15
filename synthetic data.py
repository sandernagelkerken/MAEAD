from dataloader import load_torque
from curveadaptations import *
from tqdm import tqdm
import pandas as pd
import numpy as np

def create_synthetic_dataset(combinations,
                             columns,
                             n_normal_per_task:int = 2_000,
                             n_higher_peaks:int = 25,
                             n_add_stds:int = 25,
                             n_widen_peak:int = 25,
                             n_smooth_noise:int = 25,
                             n_random_noise:int = 25,
                             n_random_point:int = 25,
                             n_shift_curve:int = 25,
                             n_reverse_curve:int = 5,
                             n_higher_variation:int = 100,
                             path:str = "C:\LocalData\Preprocessed_processed_data"):
    cols = list(columns) + ["combination_ID", "Anomaly_label"]
    normal_data = pd.DataFrame(columns= cols)
    count = 0
    print("Creating artificial normal samples")
    for shifting_type in combinations:
        for transformer_ID in combinations[shifting_type]:
            combination_ID = int(combinations[shifting_type][transformer_ID])
            df = load_torque(shifting_type, transformer_ID, path, True)[columns]
            means = df.mean(0)
            stds = df.std(0)
            cov_matrix = df.cov()
            def reduce_element(x):
                return x/3
            cov_matrix = cov_matrix.applymap(reduce_element)
            for i in tqdm(range(n_normal_per_task)):
                random_sample = np.random.multivariate_normal(means, cov_matrix)
                random_sample = [0 if i<0 else i for i in random_sample]
                random_sample = list(random_sample) + [int(combination_ID), int(0)]
                normal_data.loc[count] = random_sample
                count += 1
    anomalous_data = pd.DataFrame(columns= cols)
    count = 0
    print("Creating artificial anomalous samples")
    for shifting_type in combinations:
        for transformer_ID in combinations[shifting_type]:
            combination_ID = combinations[shifting_type][transformer_ID]
            df = load_torque(shifting_type, transformer_ID, path, True)[columns]
            means = df.mean(0)
            stds = df.std(0)
            cov_matrix = df.cov()
            def change_element(x):
                return x*5
            amplified_cov_matrix = cov_matrix.applymap(change_element)
            for i in range(n_higher_peaks):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = change_peak_height(curve, 1.5+(i/n_higher_peaks)*5)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
            for i in range(n_add_stds):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = add_standard_deviation(curve, stds, 3+(i/n_add_stds)*5)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
            for i in range(n_widen_peak):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = widen_peak(curve, 1.5+(i/n_widen_peak)*5)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
            for i in range(n_smooth_noise):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = add_smooth_noise(curve, 5+(i/n_smooth_noise)*20)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
            for i in range(n_random_noise):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = add_random_noise(curve, 5+(i/n_random_noise)*10)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
            for i in range(n_random_point):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = change_random_point(curve, 10+(i/n_random_point)*100)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
            plusminus = 1
            for i in range(n_shift_curve):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                plusminus = plusminus * -1
                try:
                    new_curve = shift_peak(curve,  plusminus*(10 +(i/n_shift_curve)*100))
                    new_curve = [0 if i<0 else i for i in new_curve]
                    anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                    count += 1
                except IndexError:
                    pass
            for i in range(n_reverse_curve):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = reverse_curve(curve)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
            for i in range(n_higher_variation):
                new_curve = np.random.multivariate_normal(means, amplified_cov_matrix)
                new_curve = [0 if i<0 else i for i in new_curve]
                for i in range(len(new_curve)):
                    if new_curve[i] < means[i]-3*stds[i]:
                        anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                        count += 1
                        break
                    else:
                        if new_curve[i] > means[i]+3*stds[i]:
                            anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                            count += 1
                            break
    synthetic_data = pd.concat([normal_data, anomalous_data])
    print("Done")
    return synthetic_data  