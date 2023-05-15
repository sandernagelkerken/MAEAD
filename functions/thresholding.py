import numpy as np
import pandas as pd
import torch
from functions.MAE import create_Vanilla_MAE, train_model, create_dataloader, create_dataset
from functions.dataloader import load_threshold, load_torque
from functions.curveadaptations import *
import matplotlib.pyplot as plt
import random
from functions.evaluation import evaluate_on_dataset

class DynamicThreshold:
    def __init__(self, alpha):
        self.alpha = alpha
        self.a = np.array([])
        self.a_s = np.array([])
        self.alarm_locations = []
        self.current_state = None

    def add_historic_data(self, data):
        data = np.array(data)
        self.a = np.append(self.a, data)
        self.a_s = np.array([])
        for i in range(len(self.a)):
            if i == 0:
                self.a_s = np.append(self.a_s, self.a[i])
            else:
                x = (1-self.alpha) * self.a_s[i-1] + self.alpha * self.a[i]
                self.a_s = np.append(self.a_s, x)
    
    def compute_threshold(self, z_values, return_z:bool = False):
        best_z = None
        best_threshold = None
        best_criterion = -1
        self.mu_s = np.mean(self.a_s)
        self.sigma_s = np.std(self.a_s)
        for z in z_values:
            threshold = self.mu_s + self.sigma_s * z
            a_a = self.a_s[self.a_s > threshold]
            a_n = self.a_s[self.a_s <= threshold]
            delta_mu_s = self.mu_s - np.mean(a_n)
            delta_sigma_s = self.sigma_s - np.std(a_n)
            n_anomalous_intervals = 0
            for i in range(len(self.a_s)):
                if self.a_s[i] > threshold:
                    if self.a_s[i-1] <= threshold:
                        n_anomalous_intervals += 1
            if (len(a_a) + n_anomalous_intervals**2) != 0:
                criterion = ((delta_mu_s/self.mu_s) + (delta_sigma_s/self.sigma_s))/(len(a_a) + n_anomalous_intervals**2)
            else:
                criterion = ((delta_mu_s/self.mu_s) + (delta_sigma_s/self.sigma_s))/0.9
            if criterion >= best_criterion:
                best_criterion = criterion
                best_threshold = threshold
                best_z = z
        self.threshold = best_threshold
        for i in np.arange(0,len(self.a_s))[self.a_s > threshold]:
            self.alarm_locations.append(i)
        if self.a_s[-1] > best_threshold:
            self.current_state = "Anomalous"
        else:
            self.current_state = "Normal"
        if return_z:
            return best_threshold, best_z
        else:
            return best_threshold
                
    def add_observation(self, a_i, print_alarms:bool = True, show_state:bool = True):
        location = int(len(self.a_s))
        x = (1-self.alpha) * self.a_s[-1] + self.alpha * a_i
        self.a_s = np.append(self.a_s, x)
        if x > self.threshold:
            self.current_state = "Anomalous"
        else:
            self.current_state = "Normal"
        if print_alarms:
            if x > self.threshold:
                self.alarm_locations.append(location)
                if location-1 not in self.alarm_locations:
                    print(f"Dynamic threshold crossed at t={location}")
            else:
                if location-1 in self.alarm_locations:
                    print(f"System returned to normal state at t={location}")
        if show_state:
            self.plot_system_state()

    def get_current_state(self):
        return self.current_state

    def get_current_score(self):
        return self.a_s[-1]

    def plot_system_state(self, n_observations:int = None):
        if n_observations == None:
            a = self.a_s
        else:
            a = self.a_s[-n_observations:] 
        def interpolate_threshold_crosses(t, x, threshold=0):
            ta = []
            positive = (x-threshold) > 0
            ti = np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
            for i in ti:
                y_ = np.sort(x[i:i+2])
                z_ = t[i:i+2][np.argsort(x[i:i+2])]
                t_ = np.interp(threshold, y_, z_)
                ta.append( t_ )
            tnew = np.append( t, np.array(ta) )
            xnew = np.append( x, np.ones(len(ta))*threshold )
            xnew = xnew[tnew.argsort()]
            tnew = np.sort(tnew)
            return tnew, xnew
        plt.plot(np.arange(0,len(a)), [self.threshold]*len(a), color = "red")
        t,x = interpolate_threshold_crosses(np.arange(0,len(a)), a, self.threshold)
        a_s = np.copy(x)
        a_a = np.copy(x)
        a_s[a_s>self.threshold] = np.nan
        a_a[a_a<=self.threshold] = np.nan
        plt.plot(t, a_s, color = 'black', linewidth = 0.5, label = 'threshold')
        plt.plot(t, a_a, color = 'red', linewidth = 0.5, label = f'smoothed anomaly score (alpha = {self.alpha})')
        plt.fill_between(np.arange(0,len(a)), a, [self.threshold]*len(a), where = (a>self.threshold), interpolate=True, facecolor = 'red', alpha = 1)
        plt.title("System state")
        plt.legend()

    def get_threshold(self):
        try:
            return self.threshold
        except NameError:
            raise RuntimeError("Compute the threshold first")

def create_dynamic_threshold(alpha):
    return DynamicThreshold(alpha)

class MAEImplementation:
    def __init__(self, combinations, alpha):
        self.combinations = combinations
        self.dynamic_thresholds = {}
        self.alpha = alpha
        self.task_IDs = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_loss_function= torch.nn.MSELoss
        self.optimizer = torch.optim.RAdam
        self.train_loss_function = torch.nn.L1Loss
        self.cols = [str(i) for i in np.arange(0,300,1)]
        for shifting_type in combinations:
            for transformer_ID in combinations[shifting_type]:
                task_ID = combinations[shifting_type][transformer_ID]
                self.task_IDs.append(task_ID)
                self.dynamic_thresholds[task_ID] = DynamicThreshold(self.alpha)
        self.historic_score = {}

    def set_anomaly_detector(self, model):
        self.model = model

    def train_anomaly_detector(self, num_epochs:int, learning_rate:float, path_to_training_data:str,
                                batch_size:int, convergence_plot:bool = True):
        dataset = create_dataset(self.combinations, self.cols, path = path_to_training_data)
        data = dataset.df_torque
        min_dataset_size = 1_000_000
        for i in self.task_IDs:
            data_i = data[data["combination_ID"] == int(i)]
            if min_dataset_size > len(data_i):
                min_dataset_size = len(data_i)
        dataloader = create_dataloader(dataset, batch_size, int(np.floor(min_dataset_size/batch_size)))
        train_model(self.model, num_epochs, dataloader, self.train_loss_function, self.optimizer, self.device,
                    len(self.cols), learning_rate, False, "", convergence_plot, 0.2, False)

    def add_historic_data(self, path):
        self.model.eval()
        loss_function = self.eval_loss_function(reduction='none')
        data = torch.Tensor(create_dataset(self.combinations, [str(i) for i in np.arange(0,300,1)], path = path).df_torque.values).to(self.device)
        for i in self.task_IDs:
            data_i = data[data[:,-1] == int(i)]
            n_chunks_i = round(data_i.shape[0]/256)
            anomaly_scores_i = torch.empty(0).to(self.device)
            for chunk in torch.tensor_split(data_i, n_chunks_i):                                    #Iterate over distinct chunks of the data
                output = self.model(chunk).data                                                     #Run the data through the model
                reconstruction_errors = torch.mean(loss_function(chunk[:,:-1], output), dim = 1)    #Compute the reconstruction errors
                anomaly_scores_i = torch.cat((anomaly_scores_i, reconstruction_errors))             #Save the reconstruction errors in the tensor
            self.dynamic_thresholds[i].add_historic_data(anomaly_scores_i)

    def compute_dynamic_thresholds(self, z_values):
        for i in self.task_IDs:
            self.dynamic_thresholds[i].compute_threshold(z_values)

    def add_sequence(self, sequence, task_ID, print_alarms:bool = True, show_state:bool = True):
        self.model.eval()
        loss_function = self.eval_loss_function()
        input = torch.concat((torch.tensor(sequence),torch.tensor([task_ID]))).float().to(self.device)
        output = self.model(input.unsqueeze(0))
        anomaly_score = loss_function(input[:-1], output).detach().numpy()
        self.dynamic_thresholds[task_ID].add_observation(anomaly_score, print_alarms, False)
        if show_state:
            self.plot_system_state(task_ID)

    def plot_system_state(self, task_ID, n_observations:int = None):
        thresholder = self.dynamic_thresholds[task_ID]
        if n_observations == None:
            a = thresholder.a_s
        else:
            a = thresholder.a_s[-n_observations:] 
        def interpolate_threshold_crosses(t, x, threshold=0):
            ta = []
            positive = (x-threshold) > 0
            ti = np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
            for i in ti:
                y_ = np.sort(x[i:i+2])
                z_ = t[i:i+2][np.argsort(x[i:i+2])]
                t_ = np.interp(threshold, y_, z_)
                ta.append( t_ )
            tnew = np.append( t, np.array(ta) )
            xnew = np.append( x, np.ones(len(ta))*threshold )
            xnew = xnew[tnew.argsort()]
            tnew = np.sort(tnew)
            return tnew, xnew
        plt.plot(np.arange(0,len(a)), [thresholder.threshold]*len(a), color = "red")
        t,x = interpolate_threshold_crosses(np.arange(0,len(a)), a, thresholder.threshold)
        a_s = np.copy(x)
        a_a = np.copy(x)
        a_s[a_s>thresholder.threshold] = np.nan
        a_a[a_a<=thresholder.threshold] = np.nan
        plt.plot(t, a_s, color = 'black', linewidth = 0.5)
        plt.plot(t, a_a, color = 'red', linewidth = 0.5, label = 'threshold')
        plt.fill_between(np.arange(0,len(a)), a, [thresholder.threshold]*len(a), where = (a>thresholder.threshold), interpolate=True, facecolor = 'red', alpha = 1)
        plt.title(f"System state of task {task_ID}")
        plt.ylabel("Anomaly score")
        plt.xlabel("Time (#observations)")
        plt.legend()
        plt.show()

    def get_system_state(self, task_ID):
        return self.dynamic_thresholds[task_ID].get_current_state()

    def get_system_score(self, task_ID):
        return self.dynamic_thresholds[task_ID].get_current_score()

    def get_threshold_value(self, task_ID):
        return self.dynamic_thresholds[task_ID].get_threshold()
    
    def get_task_most_at_risk(self, show_state:bool = True):
        max_risk_score = -1000
        for id in self.task_IDs:
            dt = self.dynamic_thresholds[id]
            risk_score = (dt.get_threshold() - dt.get_current_score()) / dt.sigma_s
            if risk_score > max_risk_score:
                max_risk_score = risk_score
                task_most_at_risk = id
        if show_state:
            self.plot_system_state(task_most_at_risk)
        return task_most_at_risk
    
    def get_shifting_type_transformer_ID(self, task_ID):
        for shifting_type in self.combinations:
            for transformer_ID in self.combinations[shifting_type]:
                if self.combinations[shifting_type][transformer_ID] == int(task_ID):
                    return shifting_type, transformer_ID
        raise IndexError(f"No task ID {task_ID}")
    
class OriginalImplementation:
    def __init__(self, combinations) -> None:
        self.combinations = combinations
        self.task_IDs = []
        self.thresholds = {}
        self.max_distances = {}
        self.system_state = {}
        for shifting_type in combinations:
            for transformer_ID in combinations[shifting_type]:
                task_ID = combinations[shifting_type][transformer_ID]
                self.task_IDs.append(task_ID)
                self.thresholds[task_ID] = load_threshold(shifting_type, transformer_ID).median(0).values
                self.max_distances[task_ID] = [-10]
                self.system_state[task_ID] = 'Normal'

    def add_sequence(self, sequence, task, print_alarms:bool = True, show_state:bool = True):
        difference = np.subtract(sequence, self.thresholds[task])
        max_difference = np.max(difference)
        self.max_distances[task].append(max_difference)
        if max_difference > 0:
            self.system_state[task] = 'Anomalous'
        else:
            self.system_state[task] = 'Normal'
            
    def get_system_state(self, task):
        return self.system_state[task]
    
    def get_system_score(self, task):
        return self.max_distances[task][-1]
    
    def get_threshold_value(self, task):
        return np.float64(0)
    
class DirectImplementation:
    def __init__(self, combinations, thresholds) -> None:
        self.combinations = combinations
        self.task_IDs = []
        self.thresholds = {}
        self.anomaly_scores = {}
        self.system_state = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_loss_function= torch.nn.MSELoss
        self.optimizer = torch.optim.RAdam
        self.train_loss_function = torch.nn.L1Loss
        self.cols = [str(i) for i in np.arange(0,300,1)]
        for shifting_type in combinations:
            for transformer_ID in combinations[shifting_type]:
                task_ID = combinations[shifting_type][transformer_ID]
                self.task_IDs.append(task_ID)
                self.anomaly_scores[task_ID] = [-10]
                self.system_state[task_ID] = 'Normal'
                self.thresholds[task_ID] = thresholds[task_ID]

    def set_anomaly_detector(self, model):
        self.model = model

    def add_sequence(self, sequence, task:int, print_alarms:bool = True, show_state:bool = True):
        self.model.eval()
        loss_function = self.eval_loss_function()
        input = torch.concat((torch.tensor(sequence),torch.tensor([task]))).float().to(self.device)
        output = self.model(input.unsqueeze(0))
        anomaly_score = loss_function(input[:-1], output).detach().numpy()
        self.anomaly_scores[task].append(anomaly_score)
        if anomaly_score > self.thresholds[task]:
            self.system_state[task] = 'Anomalous'
        else:
            self.system_state[task] = 'Normal'
        
    def get_system_state(self, task):
        return self.system_state[task]
    
    def get_system_score(self, task):
        return self.anomaly_scores[task][-1]
    
    def get_threshold_value(self, task):
        return self.thresholds[task]
    
def create_dynamic_monitor(combinations, model, alpha:float = 0.25, path:str = "C:\LocalData\Data\Completely_preprocessed_data", zs:list = np.arange(2,10,0.05)):
    dynamicmonitor = MAEImplementation(combinations, 0.25)
    dynamicmonitor.set_anomaly_detector(model)
    dynamicmonitor.add_historic_data("C:\LocalData\Data\Completely_preprocessed_data")
    dynamicmonitor.compute_dynamic_thresholds(np.arange(2,10,0.05))
    return dynamicmonitor

def compute_thresholds(model, test_dataset, fpr):
    result = evaluate_on_dataset(model, test_dataset, torch.nn.MSELoss(), shared_threshold=False)
    thresholds = {}
    for i in range(len(result[1][0])):
        j= 0
        while result[1][0][i][j] < fpr:
            j+=1
        t = result[1][2][i][j]
        thresholds[i] = t
    return thresholds

def create_static_monitor(combinations, model, thresholds):
    staticmonitor = DirectImplementation(combinations, thresholds)
    staticmonitor.set_anomaly_detector(model)
    return staticmonitor

def create_baseline_monitor(combinations):
    baselinemonitor = OriginalImplementation(combinations)
    return baselinemonitor

def simulation(monitor:MAEImplementation, task_ID:int, combinations, normal_duration:int, anomalous_onset:int, anomalous_duration:int, anomaly_type:str, max_anomalous_degree:int = 1, alternating_anomalies:bool = False, anomalous_frequency:int = 5, plot_process:bool = True, return_anomalous_states:bool = True, path:str = "C:\LocalData\Data\Completely_preprocessed_data"):
    found = False
    for s in combinations:
        for t in combinations[s]:
            if combinations[s][t] == int(task_ID):
                shifting_type = s
                transformer_ID = t
                found = True
    if found == False:
        raise IndexError(f"No task ID {task_ID} in the provided combinations")
    df = load_torque(shifting_type, transformer_ID, path, True)[[str(i) for i in np.arange(0,300,1)]]
    means = df.mean(0)
    stds = df.std(0)
    cov_matrix = df.cov()
    def create_normal_sample(means, cov_matrix):
        def reduce_element(x):
            return x/3
        cov_matrix = cov_matrix.applymap(reduce_element)
        random_sample = np.random.multivariate_normal(means, cov_matrix)
        random_sample = [0 if i<0 else i for i in random_sample]
        random_sample = list(random_sample)
        return random_sample
    def create_anomalous_sample(means, stds, cov_matrix, degree:float):
        def change_element(x):
            return x*(2+3*degree)
        normal_sample = create_normal_sample(means, cov_matrix)
        amplified_cov_matrix = cov_matrix.applymap(change_element)
        if anomaly_type == "random":
            a = ["higher peak", "add std", "widen peak", "smooth noise", "random noise", "random point", "shift curve", "higher variation"][random.randint(0,7)]
        elif anomaly_type in ["higher peak", "add std", "widen peak", "smooth noise", "random noise", "random point", "shift curve", "reverse curve", "higher variation"]:
            a = anomaly_type
        else:
            raise ValueError(f"Unknown anomaly type {anomaly_type}")
        if a == "higher peak":
            curve = normal_sample
            new_curve = change_peak_height(curve, 1+(degree)*4)
            new_curve = [0 if i<0 else i for i in new_curve]
            return list(new_curve)
        if a == "add std":
            curve = normal_sample
            new_curve = add_standard_deviation(curve, stds, 3+(degree)*5)
            new_curve = [0 if i<0 else i for i in new_curve]
            return list(new_curve)
        if a == "widen peak":
            curve = normal_sample
            new_curve = widen_peak(curve, 1.5+(degree)*5)
            new_curve = [0 if i<0 else i for i in new_curve]
            return list(new_curve)
        if a == "smooth noise":
            curve = normal_sample
            new_curve = add_smooth_noise(curve, 5+(degree)*20)
            new_curve = [0 if i<0 else i for i in new_curve]
            return list(new_curve)
        if a == "random noise":
            curve = normal_sample
            new_curve = add_random_noise(curve, 5+(degree)*10)
            new_curve = [0 if i<0 else i for i in new_curve]
            return list(new_curve)
        if a == "random point":
            curve = normal_sample
            new_curve = change_random_point(curve, 10+(degree)*100)
            new_curve = [0 if i<0 else i for i in new_curve]
            return list(new_curve)
        plusminus = 1
        if a == "shift curve":
            curve = normal_sample
            plusminus = plusminus * -1
            try:
                new_curve = shift_peak(curve,  plusminus*(10 +(degree)*100))
                new_curve = [0 if i<0 else i for i in new_curve]
                return list(new_curve)
            except IndexError:
                pass
        if a == "reverse curve":
            curve = normal_sample
            new_curve = reverse_curve(curve)
            new_curve = [0 if i<0 else i for i in new_curve]
            return list(new_curve)
        if a == "higher variation":
            new_curve = np.random.multivariate_normal(means, amplified_cov_matrix)
            new_curve = [0 if i<0 else i for i in new_curve]
            for i in range(len(new_curve)):
                if new_curve[i] < means[i]-3*stds[i]:
                    return list(new_curve)
                else:
                    if new_curve[i] > means[i]+3*stds[i]:
                        return list(new_curve)
                    else:
                        new_curve = np.random.multivariate_normal(means, amplified_cov_matrix)
                        new_curve = [0 if i<0 else i for i in new_curve]
                        for i in range(len(new_curve)):
                            if new_curve[i] < means[i]-3*stds[i]:
                                return list(new_curve)
                            else:
                                if new_curve[i] > means[i]+3*stds[i]:
                                    return list(new_curve)
                                else:
                                    new_curve = np.random.multivariate_normal(means, amplified_cov_matrix)
                                    new_curve = [0 if i<0 else i for i in new_curve]
                                    return list(new_curve)
    values = [monitor.get_system_score(task_ID)]
    anomalous_states = []
    for i in range(normal_duration):
        monitor.add_sequence(create_normal_sample(means, cov_matrix),task_ID, False, False)
        after = monitor.get_system_score(task_ID)
        values.append(after)
        anomalous_states.append(True if monitor.get_system_state(task_ID) == "Anomalous" else False)
    for i in range(normal_duration,normal_duration+anomalous_onset):
        if alternating_anomalies:
            r_anomalous = True if i % 5 == 4 else False
            if r_anomalous:
                monitor.add_sequence(create_anomalous_sample(means, stds, cov_matrix, ((i-normal_duration)/anomalous_onset)*max_anomalous_degree), task_ID, False, False)
            else:
                monitor.add_sequence(create_normal_sample(means, cov_matrix),task_ID, False, False)
        else:
            monitor.add_sequence(create_anomalous_sample(means, stds, cov_matrix, ((i-normal_duration)/anomalous_onset)*max_anomalous_degree), task_ID, False, False)
        after = monitor.get_system_score(task_ID)
        values.append(after)
        anomalous_states.append(True if monitor.get_system_state(task_ID) == "Anomalous" else False)
    for i in range(normal_duration+anomalous_onset, normal_duration+anomalous_onset+anomalous_duration):
        if alternating_anomalies:
            r_anomalous = True if i % 5 == 4 else False
            if r_anomalous:
                monitor.add_sequence(create_anomalous_sample(means, stds, cov_matrix, max_anomalous_degree), task_ID, False, False)
            else:
                monitor.add_sequence(create_normal_sample(means, cov_matrix),task_ID, False, False)
        else:
            monitor.add_sequence(create_anomalous_sample(means, stds, cov_matrix, max_anomalous_degree), task_ID, False, False)
        after = monitor.get_system_score(task_ID)
        values.append(after)
        anomalous_states.append(True if monitor.get_system_state(task_ID) == "Anomalous" else False)
    if plot_process:
        values = values[1:]
        threshold = monitor.get_threshold_value(task_ID)
        time = np.arange(0,len(values))
        def interpolate_threshold_crosses(t, x, threshold=0):
            ta = []
            positive = (x-threshold) > 0
            ti = np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
            for i in ti:
                y_ = np.sort(x[i:i+2])
                z_ = t[i:i+2][np.argsort(x[i:i+2])]
                t_ = np.interp(threshold, y_, z_)
                ta.append( t_ )
            tnew = np.append( t, np.array(ta) )
            xnew = np.append( x, np.ones(len(ta))*threshold )
            xnew = xnew[tnew.argsort()]
            tnew = np.sort(tnew)
            return tnew, xnew
        plt.plot(np.arange(0,len(time)), [threshold]*len(time), color = "red")
        t,x = interpolate_threshold_crosses(np.arange(0,len(values)), values, threshold)
        a_s = np.copy(x)
        a_a = np.copy(x)
        a_s[a_s>threshold] = np.nan
        a_a[a_a<=threshold] = np.nan
        plt.plot(t, a_s, color = 'black', linewidth = 0.5)
        plt.plot(t, a_a, color = 'red', linewidth = 0.5, label = 'threshold')
        plt.fill_between(np.arange(0,len(values)), values, [threshold]*len(values), where = (values>threshold), interpolate=True, facecolor = 'red', alpha = 1)
        plt.title(f"System state of task {task_ID}")
        plt.ylabel("Anomaly score")
        plt.xlabel("Time (#observations)")
        plt.legend()
        plt.show()
    if return_anomalous_states:
        return anomalous_states