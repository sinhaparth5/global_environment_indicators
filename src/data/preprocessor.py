import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class EnvironmentalDataPreprocessing:
    def __init__(self):
        self.scalers = {}
        
    def preprocess_time_series(self, df, target_column, time_column='Year'):
        """ Preprocess time series data for the model. """
        # Ensure data is sorted by time
        df = df.sort_values(by=time_column)
        
        # Handle missing values
        df[target_column] = df[target_column].interpolate(method='linear')
        
        # Scale the target variable
        scaler = MinMaxScaler()
        df[f'{target_column}_scaled'] = scaler.fit_transform(df[[target_column]])
        self.scalers[target_column] = scaler
        
        return df
    
    def prepare_sequences(self, data, seq_length):
        """ Prepare sequences for time series prediction. """
        sequences = []
        targets = []
        
        for i in range(self, data, seq_length):
            seq = data[i:i + seq_length]
            target = data[i + seq_length]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)