# src/data/data_loader.py
import pandas as pd
import yaml
from pathlib import Path
import os

class EnvironmentalDataLoader:
    def __init__(self, config_path='config.yaml'):
        try:
            # Correct path resolution (3 levels up from src/data)
            self.project_root = Path(__file__).resolve().parent.parent.parent
            self.config = self._load_config(config_path)
            
            self.raw_data_path = self.project_root / self.config['data_paths']['raw_data']
            self.categories = self.config['data_categories']
            
            print(f"Project root: {self.project_root}")
            print(f"Raw data path: {self.raw_data_path}")

        except Exception as e:
            print(f"Loader initialization failed: {str(e)}")
            raise

    def _load_config(self, config_path):
        """Load configuration from project root"""
        config_file = self.project_root / config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def process_raw_csv(self, file_path: Path) -> pd.DataFrame:
        """Clean and standardize raw CSV data"""
        try:
            df = pd.read_csv(file_path)
            
            # Clean column names
            df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
            
            # Convert numeric columns
            for col in df.columns[2:]:  # Skip first 2 ID columns
                if df[col].dtype == object:
                    df[col] = df[col].replace(['...', '', ' ', 'NA'], pd.NA)
                    df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna(how='all')

        except Exception as e:
            print(f"CSV processing failed for {file_path.name}: {str(e)}")
            return pd.DataFrame()

    def load_category(self, category_name: str):
        """Load all datasets for a specific category"""
        try:
            if category_name not in self.categories:
                raise ValueError(f"Invalid category. Available: {self.categories}")
            
            category_dir = self.raw_data_path / category_name
            if not category_dir.exists():
                raise FileNotFoundError(f"Category directory missing: {category_dir}")
            
            datasets = {}
            for csv_file in category_dir.glob('*.csv'):
                df = self.process_raw_csv(csv_file)
                if not df.empty:
                    datasets[csv_file.stem] = df
                    print(f"Loaded {csv_file.stem} with shape {df.shape}")
            
            return datasets

        except Exception as e:
            print(f"Category load failed for {category_name}: {str(e)}")
            raise