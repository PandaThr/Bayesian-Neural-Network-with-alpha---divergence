import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
import datetime
def plot_av_reward(data, confidence=0.95, x_label=None, y_label=None, title=None,bar_alpha = 0.2,label="",color="k"):
     # Calculate the mean and standard error over time (episodes)
    average = np.mean(data, axis=0)
    n = data.shape[0]
    stderr = np.std(data, axis=0) / np.sqrt(n)
    margin = stderr * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)

     # Calculate upper and lower bounds of the confidence interval
    lower_bound = average - margin
    upper_bound = average + margin

    plt.plot(range(data.shape[1]), average, label=label,color=color)
    plt.fill_between(range(data.shape[1]), lower_bound, upper_bound, alpha=bar_alpha,color=color)
    
    
    if x_label:
        plt.xlabel(x_label)
    
    if y_label:
        plt.ylabel(y_label)
    
    if title:
        plt.title(title)
    
    plt.grid()
def load_function(str):
        with open(str,'rb') as f:
            res = pickle.load(f)
        return res
def save_function(data, base_name,data_prompt = 1):
    file_name = f"{base_name}"
    if data_prompt:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{base_name}__{formatted_datetime}"
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)