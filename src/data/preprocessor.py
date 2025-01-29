# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import yaml

class EnvironmentalDataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        try:
            # Path configuration
            self.project_root = Path(__file__).resolve().parent.parent
            self.config = self._load_config(config_path)
            self.processed_path = self.project_root / self.config['data_paths']['processed_data']
            self.scalers = {}
            
            # Create processed directory if needed
            self.processed_path.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            print(f"Preprocessor initialization failed: {str(e)}")
            raise

    def _load_config(self, config_path):
        """Load configuration file"""
        config_file = self.project_root / config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def preprocess_timeseries(self, df: pd.DataFrame, target_col: str, time_col: str = 'year'):
        """Process a time series dataframe"""
        try:
            # Convert and sort time
            df[time_col] = pd.to_datetime(df[time_col], format='%Y').dt.year
            df = df.sort_values(time_col).reset_index(drop=True)
            
            # Handle missing values
            df[target_col] = df[target_col].interpolate(method='linear')
            
            # Scale features
            scaler = MinMaxScaler()
            df[f'{target_col}_scaled'] = scaler.fit_transform(df[[target_col]])
            self.scalers[target_col] = scaler
            
            return df.dropna()

        except Exception as e:
            print(f"Timeseries preprocessing failed: {str(e)}")
            raise

    def create_sequences(self, data: np.array, seq_length: int):
        """Create time series sequences for training"""
        try:
            sequences = []
            targets = []
            
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i+seq_length])
                targets.append(data[i+seq_length])
                
            return np.array(sequences), np.array(targets)

        except Exception as e:
            print(f"Sequence creation failed: {str(e)}")
            raise

    def save_processed_data(self, df: pd.DataFrame, category: str, filename: str):
        """Save processed data to Parquet format"""
        try:
            save_dir = self.processed_path / category
            save_dir.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(save_dir / f"{filename}.parquet")
            print(f"Saved processed data to {save_dir / filename}")

        except Exception as e:
            print(f"Data save failed: {str(e)}")
            raise