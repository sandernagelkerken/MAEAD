import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from tqdm import tqdm 
from functions.dataloader import load_torque
from functions.multitaskAE import create_dataloader, train_model
from functions.combinations import get_n_combinations, get_closest_combinations
from functions.curveadaptations import *
import numpy as np
from datetime import datetime
import pickle

def create_synthetic_dataset(combinations,
                             columns,
                             n_anomalies_per_type_per_task:int = None,
                             n_normal_per_task:int = None,
                             n_higher_peaks:int = 25,
                             n_add_stds:int = 25,
                             n_widen_peak:int = 25,
                             n_smooth_noise:int = 25,
                             n_random_noise:int = 25,
                             n_random_point:int = 25,
                             n_shift_curve:int = 25,
                             n_reverse_curve:int = 25,
                             n_higher_variation:int = 25,
                             show_tqdm:bool = True,
                             path:str = "C:/LocalData/Data/Completely_preprocessed_data"):
    if n_anomalies_per_type_per_task != None:
        n_higher_peaks = n_anomalies_per_type_per_task
        n_add_stds = n_anomalies_per_type_per_task
        n_widen_peak = n_anomalies_per_type_per_task
        n_smooth_noise = n_anomalies_per_type_per_task
        n_random_noise = n_anomalies_per_type_per_task
        n_random_point = n_anomalies_per_type_per_task
        n_shift_curve = n_anomalies_per_type_per_task
        n_reverse_curve = n_anomalies_per_type_per_task
        n_higher_variation = n_anomalies_per_type_per_task
    if n_normal_per_task == None:                                               #Introduce 10% anomalies at default
        n_normal_per_task = 9*(n_higher_peaks + n_add_stds + n_widen_peak + n_smooth_noise + n_random_noise + n_random_point + n_shift_curve + n_reverse_curve + n_higher_variation)
    cols = list(columns) + ["combination_ID", "Anomaly_label"]
    normal_data = pd.DataFrame(columns= cols)
    count = 0
    if show_tqdm:
        pb = tqdm(total=n_normal_per_task*get_n_combinations(combinations), desc="Creating normal data", leave=False)
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
            for i in range(n_normal_per_task):
                random_sample = np.random.multivariate_normal(means, cov_matrix)
                random_sample = [0 if i<0 else i for i in random_sample]
                random_sample = list(random_sample) + [int(combination_ID), int(0)]
                normal_data.loc[count] = random_sample
                count += 1
                if show_tqdm:
                    pb.update(1)
    if show_tqdm:
        pb.close()
    anomalous_data = pd.DataFrame(columns= cols)
    count = 0
    if show_tqdm:
        pb = tqdm(total = get_n_combinations(combinations)*n_normal_per_task/9, desc="Creating anomalous data", leave = False)
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
                if show_tqdm:
                    pb.update(1)
            for i in range(n_add_stds):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = add_standard_deviation(curve, stds, 3+(i/n_add_stds)*5)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
                if show_tqdm:
                    pb.update(1)
            for i in range(n_widen_peak):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = widen_peak(curve, 1.5+(i/n_widen_peak)*5)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
                if show_tqdm:
                    pb.update(1)
            for i in range(n_smooth_noise):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = add_smooth_noise(curve, 5+(i/n_smooth_noise)*20)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
                if show_tqdm:
                    pb.update(1)
            for i in range(n_random_noise):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = add_random_noise(curve, 5+(i/n_random_noise)*10)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
                if show_tqdm:
                    pb.update(1)
            for i in range(n_random_point):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = change_random_point(curve, 10+(i/n_random_point)*100)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
                if show_tqdm:
                    pb.update(1)
            plusminus = 1
            for i in range(n_shift_curve):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                plusminus = plusminus * -1
                try:
                    new_curve = shift_peak(curve,  plusminus*(10 +(i/n_shift_curve)*100))
                    new_curve = [0 if i<0 else i for i in new_curve]
                    anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                    count += 1
                    if show_tqdm:
                        pb.update(1)
                except IndexError:
                    pass
            for i in range(n_reverse_curve):
                curve = list(normal_data[normal_data["combination_ID"] == combination_ID].sample().values)[0][:-2]
                new_curve = reverse_curve(curve)
                new_curve = [0 if i<0 else i for i in new_curve]
                anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                count += 1
                if show_tqdm:
                    pb.update(1)
            for i in range(n_higher_variation):
                new_curve = np.random.multivariate_normal(means, amplified_cov_matrix)
                new_curve = [0 if i<0 else i for i in new_curve]
                for i in range(len(new_curve)):
                    if new_curve[i] < means[i]-3*stds[i]:
                        anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                        count += 1
                        if show_tqdm:
                            pb.update(1)
                        break
                    else:
                        if new_curve[i] > means[i]+3*stds[i]:
                            anomalous_data.loc[count] = list(new_curve) + [combination_ID, 1]
                            count += 1
                            if show_tqdm:
                                pb.update(1)
                            break
    synthetic_data = pd.concat([normal_data, anomalous_data])
    if show_tqdm:
        pb.close()
    return synthetic_data

def evaluate_on_dataset(model,
                   data,
                   loss_function,
                   show_roc_curve:bool = True,
                   shared_threshold:bool = True,
                   AUROC:bool = True):
    model.eval()
    if shared_threshold:
        reconstruction_errors = []
        for index in range(len(data)):
            row = torch.Tensor(data.iloc[index][:-1]).unsqueeze(0)
            output = model(row).data
            reconstruction_error = loss_function(row.data[:,:-1], output)
            reconstruction_errors.append(reconstruction_error)
        fpr, tpr, thresholds = metrics.roc_curve(data["Anomaly_label"], reconstruction_errors)
        auc = metrics.auc(fpr, tpr)
        if show_roc_curve:
            plt.plot(fpr,tpr, color = 'black', label = f"ROC curve (area = {auc}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc = "lower right")
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.show()
        return auc, (fpr, tpr, thresholds)
    else:
        ids = data["combination_ID"].unique()
        aucs = {}
        fprs = {}
        tprs = {}
        precisions = {}
        recalls = {}
        thresholdss = {}
        for id in ids:
            reconstruction_errors=[]
            data_id = data[data["combination_ID"] == id]
            for index in range(len(data_id)):
                row = torch.Tensor(data_id.iloc[index][:-1]).unsqueeze(0)
                output = model(row).data
                reconstruction_error = loss_function(row.data[:,:-1], output)
                reconstruction_errors.append(reconstruction_error)
            if AUROC:
                fpr,tpr,thresholds = metrics.roc_curve(data_id["Anomaly_label"], reconstruction_errors)
                auc = metrics.auc(fpr,tpr)
                fprs[id] = fpr
                tprs[id] = tpr
            else:
                precision, recall, thresholds = metrics.precision_recall_curve(data_id["Anomaly_label"], reconstruction_errors)
                auc = metrics.auc(recall, precision)
                precisions[id] = precision
                recalls[id] = recall
            aucs[id] = auc
            thresholdss[id] = thresholds
            if show_roc_curve:
                if AUROC:
                    plt.plot(fpr,tpr, label = f"Task {id} (area = {auc}")
                else:
                    plt.plot(recall,precision, label = f"Task {id} (area = {auc}")
        avg_auc = sum(aucs.values())/len(aucs.values())
        if show_roc_curve:
            plt.plot([0,1], [0,0], color = 'white', label = f"Average area = {avg_auc}")
            if AUROC:
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend(loc = "lower right")
            else:
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.legend()
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
        if AUROC:
            return avg_auc, (fprs, tprs, thresholdss)
        else:
            return avg_auc, (recalls, precisions, thresholdss)

def evaluate_model_on_separate_anomaly_types(model, combinations, columns, loss_function, n_anomalies_per_type:int = 25, plot_results:bool = True):
    model.eval()
    scores = {}
    scores["higher \n peak"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, n_anomalies_per_type, 0, 0, 0, 0, 0, 0, 0, 0, False), loss_function, False)
    scores["higher \n curve"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, n_anomalies_per_type, 0, 0, 0, 0, 0, 0, 0, False), loss_function, False)
    scores["wider \n peak"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, 0, n_anomalies_per_type, 0, 0, 0, 0, 0, 0, False), loss_function, False)
    scores["added \n smooth \n noise"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, 0, 0, n_anomalies_per_type, 0, 0, 0, 0, 0, False), loss_function, False)
    scores["added \n random \n noise"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, n_anomalies_per_type, 0, 0, 0, 0, False), loss_function, False)
    scores["random \n point \n permutation"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, n_anomalies_per_type, 0, 0, 0, False), loss_function, False)
    scores["shifted \n curve"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, 0, n_anomalies_per_type, 0, 0, False), loss_function, False)
    scores["reversed \n curve"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, 0, 0, n_anomalies_per_type, 0, False), loss_function, False)
    scores["increased \n variation"] = evaluate_on_dataset(model, create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, 0, 0, 0, n_anomalies_per_type, False), loss_function, False)
    if plot_results:
        x = np.arange(0, 20*len(scores.keys()), 20)
        barchart = plt.bar(x, [round(i,2) for i in scores.values()], 10, color = 'black')
        plt.ylabel("AUROC")
        plt.ylim(top = 1)
        plt.xticks(x, scores.keys(), fontsize = 'x-small')
        plt.bar_label(barchart, padding=3)
        plt.show()
    return scores

def evaluate_model(model, combinations, columns, loss_function, n_anomalies_per_type:int = 25, plot_results:bool = True, shared_threshold:bool = True):
    model.eval()
    scores = {}
    data1 = create_synthetic_dataset(combinations, columns, None, n_anomalies_per_type, 0, 0, 0, 0, 0, 0, 0, 0, False)
    scores["higher \n peak"] = evaluate_on_dataset(model, data1, loss_function, False, False)[0]
    data2 = create_synthetic_dataset(combinations, columns, None, 0, n_anomalies_per_type, 0, 0, 0, 0, 0, 0, 0, False)
    scores["higher \n curve"] = evaluate_on_dataset(model, data2, loss_function, False, False)[0]
    data3 = create_synthetic_dataset(combinations, columns, None, 0, 0, n_anomalies_per_type, 0, 0, 0, 0, 0, 0, False)
    scores["wider \n peak"] = evaluate_on_dataset(model, data3, loss_function, False, False)[0]
    data4 = create_synthetic_dataset(combinations, columns, None, 0, 0, 0, n_anomalies_per_type, 0, 0, 0, 0, 0, False)
    scores["added \n smooth \n noise"] = evaluate_on_dataset(model, data4, loss_function, False, False)[0]
    data5 = create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, n_anomalies_per_type, 0, 0, 0, 0, False)
    scores["added \n random \n noise"] = evaluate_on_dataset(model, data5, loss_function, False, False)[0]
    data6 = create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, n_anomalies_per_type, 0, 0, 0, False)
    scores["random \n point \n permutation"] = evaluate_on_dataset(model, data6, loss_function, False, False)[0]
    data7 = create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, 0, n_anomalies_per_type, 0, 0, False)
    scores["shifted \n peak"] = evaluate_on_dataset(model, data7, loss_function, False, False)[0]
    data8 = create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, 0, 0, n_anomalies_per_type, 0, False)
    scores["reversed \n curve"] = evaluate_on_dataset(model, data8, loss_function, False, False)[0]
    data9 = create_synthetic_dataset(combinations, columns, None, 0, 0, 0, 0, 0, 0, 0, 0, n_anomalies_per_type, False)
    scores["increased \n variation"] = evaluate_on_dataset(model, data9, loss_function, False, False)[0]
    data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9], ignore_index=True)
    if shared_threshold:
        auc, (fpr, tpr) = evaluate_on_dataset(model, data, loss_function, False, True)
    else:
        avg_auc, (fprs, tprs) = evaluate_on_dataset(model, data, loss_function, False, False)
    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1,2, sharey = True, figsize = (20,8))
        if shared_threshold:
            ax1.plot(fpr,tpr, color = 'black', label = f"ROC curve (area = {round(auc,4)}")
        else:
            ax1.plot([0,1],[0,0], linewidth = 0, label = f"Average AUROC = {round(avg_auc,4)}")
            for id in fprs:
                ax1.plot(fprs[id],tprs[id], label = f"Task {int(id)}")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc = "lower right")
        ax1.set_xlim([0.0, 1.05])
        ax1.set_ylim([0.0, 1.05])
        x = np.arange(0, 20*len(scores.keys()), 20)
        barchart = ax2.bar(x, [round(i,2) for i in scores.values()], 10, color = 'black')
        ax2.set_ylabel("Average AUROC")
        ax2.set_xticks(x, scores.keys()) #, fontsize = 'x-small'
        ax2.bar_label(barchart, padding=3)
    plt.show()
    if shared_threshold:
        return auc, scores
    else:
        return avg_auc, scores

def save_to_run_log(model,
                    evaluation_loss_function,
                    auroc,
                    scores,
                    training_loss_function,
                    optimizer,
                    learning_rate,
                    batch_size,
                    maximum_batches_per_task,
                    epochs):
    date = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    old_run_log = pd.read_excel("C:/LocalData/Code/Thesis/run_log.xlsx")
    run_log = old_run_log.copy()
    model_name = model.__class__
    encoding_size = model.encoding_size
    combinations = model.combinations
    new_entry = [date, model_name, combinations, encoding_size, training_loss_function, evaluation_loss_function,
                 optimizer, learning_rate, batch_size, maximum_batches_per_task, epochs, auroc] + list(scores.values())
    run_log.loc[len(run_log)] = new_entry
    run_log.to_excel("C:/LocalData/Code/Thesis/run_log.xlsx", index=False)
    return old_run_log, run_log

def grid_search_evaluation(models:dict, datasets:dict, learning_rates:list, combinations, 
                            batch_size, max_batches_per_ID, train_loss_function, optimizer,
                            device, sequence_length, eval_loss_function, num_epochs, cols,
                            to_run_log:bool = False, plot_results = True):
    grid = {}
    test_data = create_synthetic_dataset(combinations, cols, show_tqdm=False,
                                        n_higher_peaks = 50,
                                        n_add_stds = 50,
                                        n_widen_peak = 50,
                                        n_smooth_noise = 50,
                                        n_random_noise = 50,
                                        n_random_point = 50,
                                        n_shift_curve = 50,
                                        n_reverse_curve = 50,
                                        n_higher_variation = 50)
    for model in models:
        grid[model] = {}
        for dataset in datasets:
            grid[model][dataset] = {}
            for lr in learning_rates:
                m = models[model]
                d = datasets[dataset]
                train_loader = create_dataloader(d, batch_size, max_batches_per_ID)     
                train_model(m, num_epochs, train_loader, train_loss_function, optimizer, device, sequence_length, lr, convergence_plot=False)
                auc, (fpr, tpr) = evaluate_on_dataset(m, test_data, eval_loss_function(), False)
                grid[model][dataset][lr] = auc
                if to_run_log:
                    save_to_run_log(m, eval_loss_function, auc, [None, None, None, None, None, None, None, None, None], train_loss_function, optimizer, lr, batch_size, max_batches_per_ID, num_epochs)
    min_auc, max_auc = 1,0
    for model in models:
        for dataset in datasets:
            for lr in learning_rates:
                if grid[model][dataset][lr] < min_auc:
                    min_auc = grid[model][dataset][lr]
                if grid[model][dataset][lr] > max_auc:
                    max_auc = grid[model][dataset][lr]
                    max_model = model
                    max_dataset = dataset
                    max_lr = lr
    print(f"The best performance was achieved by {max_model} on dataset {max_dataset} with learning rate {max_lr}: {max_auc}")
    if plot_results:
        if len(datasets) == 1:
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize = (5,4))
        else:
            fig, axs = plt.subplots(int(np.ceil(len(datasets)/2)), 2, sharex=True, sharey=True, figsize = (10,8))
        norm = matplotlib.colors.Normalize(min_auc, max_auc)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"), ax = axs, label = "AUROC")
        if len(datasets) == 1:
            matrix = []
            for model in models:
                row = []
                for lr in learning_rates:
                    row.append(grid[model][dataset][lr])
                matrix.append(row)
            ax.set_title(str(dataset))
            ax.set_xticks(np.arange(len(learning_rates)), labels = [str(i) for i in learning_rates])
            ax.set_yticks(np.arange(len(models)), labels = models.keys())
            ax.imshow(matrix, norm = norm)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                        rotation_mode="anchor")
            ax.set_xticks(np.arange(len(matrix[0])+1)-.5, minor=True)
            ax.set_yticks(np.arange(len(matrix)+1)-.5, minor=True)
            ax.grid(visible = True, which="minor", color="white", linestyle='-', linewidth=1) 
        else:
            for n, dataset in enumerate(datasets):
                matrix = []
                for model in models:
                    row = []
                    for lr in learning_rates:
                        row.append(grid[model][dataset][lr])
                    matrix.append(row)
                i = int(np.floor(n/2))
                j = n%2
                axs[i][j].set_title(str(dataset))
                axs[i][j].set_xticks(np.arange(len(learning_rates)), labels = [str(i) for i in learning_rates])
                axs[i][j].set_yticks(np.arange(len(models)), labels = models.keys())
                axs[i][j].imshow(matrix, norm = norm)
                plt.setp(axs[i][j].get_xticklabels(), rotation=45, ha="right",
                            rotation_mode="anchor")
                axs[i][j].set_xticks(np.arange(len(matrix[0])+1)-.5, minor=True)
                axs[i][j].set_yticks(np.arange(len(matrix)+1)-.5, minor=True)
                axs[i][j].grid(visible = True, which="minor", color="white", linestyle='-', linewidth=1)
                for m in range(len(models)):
                    for l in range(len(learning_rates)):
                        text = axs[i][j].text(l, m, round(matrix[m][l],3),
                                    ha="center", va="center", color="red")
    return grid

def create_evaluation_data(n_combinations:int, n_anomalies_per_type:int, pickle_data:bool = True, pickle_location:str = "evaluation_data", return_data:bool = False, pickle_combinations:bool = False):
    cols = [str(i) for i in np.arange(0,300,1)]
    validation_combinations, test_combinations = get_closest_combinations(3000, n_combinations*2, True)
    if pickle_combinations:
        with open(f"{pickle_location}\Validation_combinations_{n_combinations}.pkl", "wb") as file:
            pickle.dump(validation_combinations, file)
        with open(f"{pickle_location}\Test_combinations_{n_combinations}.pkl", "wb") as file:
            pickle.dump(test_combinations, file)
    validation_data = create_synthetic_dataset(validation_combinations, cols, n_anomalies_per_type)
    test_data = create_synthetic_dataset(test_combinations, cols, n_anomalies_per_type)
    if pickle_data:
        with open(f"{pickle_location}\Validation_data_{n_combinations}_{n_anomalies_per_type}.pkl", "wb") as file:
            pickle.dump(validation_data, file)
        with open(f"{pickle_location}\Test_data_{n_combinations}_{n_anomalies_per_type}.pkl", "wb") as file:
            pickle.dump(test_data, file)
    if return_data:
        return validation_data, test_data