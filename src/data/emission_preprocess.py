import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class EmissionsPreprocessor:
    def __init__(self):
        self.time_series_mappings = {
            'CO2': ['country_id', 'country', 'time_series_co2'],
            'CH4': ['country_id', 'country', 'time_series_ch4'],
            'N2O': ['country_id', 'country', 'time_series_n2o'],
            'NOx': ['country_id', 'country', 'time_series_nox'],
            'SO2': ['country_id', 'country', 'time_series_so2'],
            'GHG': ['country_id', 'country', 'time_series_ghg']
        }

    def clean_emissions_data(self, df: pd.DataFrame, emission_type: str) -> pd.DataFrame:
        try:
            df = df.copy()
            print(f"\nProcessing {emission_type} emissions")
            
            # Check for required columns
            if 'country_id' not in df.columns or 'country' not in df.columns:
                print(f"Missing country columns in {emission_type} data")
                return pd.DataFrame()
                
            # Identify year columns (1900-2100)
            year_columns = [col for col in df.columns 
                          if col.isdigit() and 1900 <= int(col) <= 2100]
            
            if not year_columns:
                print(f"No year columns found in {emission_type} data")
                return pd.DataFrame()
                
            # Convert year columns to numeric
            for col in year_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Create cleaned dataset
            keep_columns = ['country_id', 'country'] + year_columns
            cleaned_df = df[keep_columns].copy()
            cleaned_df['emission_type'] = emission_type
            
            return cleaned_df
            
        except Exception as e:
            print(f"Error cleaning {emission_type} data: {str(e)}")
            return pd.DataFrame()

    def preprocess_ghg_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            sectors_df = df.copy()
            
            # Process numeric columns
            for col in sectors_df.columns:
                if sectors_df[col].dtype == object:
                    sectors_df[col] = sectors_df[col].astype(str).str.replace(',', '')
                sectors_df[col] = pd.to_numeric(sectors_df[col], errors='coerce')
            
            return sectors_df.dropna(how='all')
            
        except Exception as e:
            print(f"GHG sector error: {str(e)}")
            return pd.DataFrame()

    def combine_emissions_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        emissions_dfs = []
        
        for filename, df in data_dict.items():
            try:
                # Skip sector data for now
                if 'sector' in filename.lower():
                    continue
                    
                emission_type = filename.split('_')[0].upper()
                cleaned_df = self.clean_emissions_data(df, emission_type)
                if not cleaned_df.empty:
                    emissions_dfs.append(cleaned_df)
                    print(f"Processed {filename} successfully")
                    
            except Exception as e:
                print(f"Skipping {filename}: {str(e)}")
                continue
        
        combined_emissions = pd.concat(emissions_dfs, ignore_index=True) if emissions_dfs else pd.DataFrame()
        
        # Process sector data separately
        sectors_df = pd.DataFrame()
        if 'GHG_Emissions_by_Sector' in data_dict:
            sectors_df = self.preprocess_ghg_sectors(data_dict['GHG_Emissions_by_Sector'])
        
        return combined_emissions, sectors_df

    def create_analysis_ready_dataset(self, combined_emissions: pd.DataFrame, sectors_df: pd.DataFrame) -> pd.DataFrame:
        try:
            if combined_emissions.empty:
                return pd.DataFrame()
                
            # Melt to long format
            id_vars = ['country_id', 'country', 'emission_type']
            year_cols = [col for col in combined_emissions.columns if col.isdigit()]
            
            melted = pd.melt(
                combined_emissions,
                id_vars=id_vars,
                value_vars=year_cols,
                var_name='year',
                value_name='emission_value'
            )
            
            melted['year'] = pd.to_numeric(melted['year'])
            return melted.sort_values(['country', 'emission_type', 'year'])
            
        except Exception as e:
            print(f"Analysis dataset error: {str(e)}")
            return pd.DataFrame()