import pandas as pd
import os
from pathlib import Path
import yaml

class EnvironmentalDataLoader:
    def __init__(self, config_path=None):
        # Find the project root directory
        if config_path is None:
            current_file = Path(__file__).resolve()  # Get the path of data_loader.py
            project_root = current_file.parent.parent.parent  # Go up to project root
            config_path = project_root / 'config.yaml'
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Convert raw_data path to absolute path
            project_root = Path(config_path).parent
            self.data_path = project_root / self.config['data_paths']['raw_data']
            self.categories = self.config['data_categories']
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                "Make sure config.yaml exists in the project root directory."
            )
        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")
    
    def load_category_data(self, category):
        """Load data for a specific environmental category."""
        if category not in self.categories:
            raise ValueError(f"Category {category} not found in config. Available categories: {self.categories}")
        
        category_path = self.data_path / category
        if not category_path.exists():
            raise FileNotFoundError(
                f"Directory not found: {category_path}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Data path: {self.data_path}"
            )
            
        data_frames = {}
        
        # Handle both CSV and Excel files
        for extension in ['*.csv', '*.xlsx']:
            for file in category_path.glob(extension):
                try:
                    if file.suffix == '.csv':
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    data_frames[file.stem] = df
                except Exception as e:
                    print(f"Error reading file {file}: {str(e)}")
                    continue
        
        if not data_frames:
            print(f"Warning: No data files found in {category_path}")
            
        return data_frames
    
    def load_all_data(self):
        """Load data from all categories."""
        all_data = {}
        for category in self.categories:
            try:
                all_data[category] = self.load_category_data(category)
            except Exception as e:
                print(f"Error loading category {category}: {str(e)}")
                continue
        return all_data