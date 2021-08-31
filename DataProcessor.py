import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.stattools import acf

class CreateProcessor:
    
    def __init__(self):
        self.lmbda = 0
        self.seasonal_lag = 12
        self.trend_lag = 1
        self.minv = 0
        self.maxv = 0
        self.seasonal_pattern_exist = False
        
    def process(self, series, look_back, use_boxcox=False, check_seasonal=True):
        power_transformed_series, self.lmbda = power_transform(series, use_boxcox)
        if check_seasonal and check_acf_signif(power_transformed_series, self.seasonal_lag) is False:
            self.seasonal_pattern_exist = True
            deseasonalised_series = difference(power_transformed_series, self.seasonal_lag)
        else:
            deseasonalised_series = power_transformed_series      
        detrended_series = difference(deseasonalised_series, self.trend_lag)
        normalised_series, self.minv, self.maxv = normalisation(detrended_series)
        X, y = create_supervised_dataset(normalised_series, num_in=look_back)
        return normalised_series, X, y
        
        
# Power transform (log transform if lmbda = 0, box-cox transform if lmbda > 0)
def power_transform(data, use_boxcox=False):
    # Power transform requires input data to be positive
    data = [value+1 for value in data]
    # Box-cox transform
    if use_boxcox:
        return boxcox(data)
    # Log transform
    return boxcox(data, lmbda=0), 0


# Inverse power transform
def invert_power_transform(data, lmbda):
    # Inverse log transform
    if lmbda == 0:
        return np.exp(data) - 1
    # Inverse box-cox transform
    return np.exp(np.log(lmbda * data + 1) / lmbda) - 1


# Lag-difference
def difference(data, lag):
    return [data[i] - data[i - lag] for i in range(lag, len(data))]


# Inverse lag-difference
def invert_difference(origin_data, processed_data, lag):
    return [processed_data[i-lag] + origin_data[i-lag] for i in range(lag, len(origin_data))]


# Min-max normalisation
def normalisation(data, minv=float('nan'), maxv=float('nan')):
    # Scale data into (0,1)
    if np.isnan(minv) or np.isnan(maxv):
        data = np.array(data)
        minv, maxv = data.min(), data.max()
    data = (data - minv) / (maxv - minv)
    return data, minv, maxv


# Inverse min-max normalisation
def invert_normalisation(processed_data, minv, maxv):
    return processed_data * (maxv - minv) + minv

# Check whether the autocorrelation coefficient at lag 12 is significant
def check_acf_signif(data, lag_no=12):
    acf_value = acf(data, fft=False, nlags=lag_no)[-1]
    upper_conf_level = 1.96/np.sqrt(len(data))
    lower_conf_level = -1.96/np.sqrt(len(data))
    if acf_value <= upper_conf_level and acf_value >= lower_conf_level:
        return True
    return False

# Create dataset for supervised learning
def create_supervised_dataset(data, num_in=1, num_out=1):
    data = np.array(data)
    if len(data.shape) == 1 or data.shape[1] != 1:
        data = data.reshape(-1, 1)
    X = np.roll(data, num_in)
    for i in range(num_in-1, 0, -1):
        X = np.concatenate([X, np.roll(data, i)], axis=1)
    y = data
    for i in range(1, num_out):
        y = np.concatenate([y, np.roll(data, -i)], axis=1)
    for i in range(num_in):
        X = np.delete(X, 0, 0)
        y = np.delete(y, 0, 0)
    for i in range(num_out-1):
        X = np.delete(X, -1, 0)
        y = np.delete(y, -1, 0)
    return X, y