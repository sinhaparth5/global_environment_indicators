import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class EmissionsPreprocessor:
    def __init__(self):
        # Updated column mappings based on your CSV structure
        self.time_series_mappings = {
            'CO2': ['Country ID', 'Country', 'Time Series - CO₂ total emissions  without LULUCF, in 1000 t'],
            'CH4': ['Country ID', 'Country', 'Time Series - Total CH4 Emissions, in 1000 tonnes of CO₂ equivalent'],
            'N2O': ['Country ID', 'Country', 'Time series - Total N2O emissions, in 1000 tonnes of CO2 equivalent'],
            'NOx': ['Country ID', 'Country', 'Time Series - NOx total emissions, in 1000 t'],
            'SO2': ['Country ID', 'Country', 'Time Series - SO2 total emissions, in 1000 t'],
            'GHG': ['Country ID', 'Country', 'Time Series - Greenhouse Gas Emissions (GHG) total without LULUCF, in 1000 tonnes of CO2 equivalent']
        }

    def clean_emissions_data(self, df: pd.DataFrame, emission_type: str) -> pd.DataFrame:
        """Clean individual emissions dataset with improved error handling."""
        try:
            # Make a copy to avoid modifying original data
            df = df.copy()
            
            # Print debugging information
            print(f"\nProcessing {emission_type} emissions")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Ensure required columns exist
            if 'Country ID' not in df.columns or 'Country' not in df.columns:
                print(f"Required columns missing in {emission_type} dataset")
                print(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()  # Return empty DataFrame if required columns missing
            
            # Clean country data if column exists
            df['Country'] = df['Country'].astype(str).str.strip()
            
            # Extract year columns (they are typically unnamed)
            year_columns = []
            for col in df.columns:
                if col.startswith('Unnamed:'):
                    # Try to convert to numeric and check if it contains valid data
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_data.isna().all():  # If column contains some numeric data
                        year_columns.append(col)
                        df[col] = numeric_data
            
            # Create cleaned dataset
            keep_columns = ['Country ID', 'Country'] + year_columns
            cleaned_df = df[keep_columns].copy()
            
            # Rename year columns (assuming they represent years from 1990 onwards)
            year_mapping = dict(zip(year_columns, range(1990, 1990 + len(year_columns))))
            cleaned_df = cleaned_df.rename(columns=year_mapping)
            
            # Add emission type identifier
            cleaned_df['emission_type'] = emission_type
            
            return cleaned_df
            
        except Exception as e:
            print(f"Error in clean_emissions_data for {emission_type}: {str(e)}")
            return pd.DataFrame()

    def preprocess_ghg_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess GHG emissions by sector data with improved error handling."""
        try:
            sectors_df = df.copy()
            
            # Identify and convert percentage columns
            percentage_cols = [col for col in sectors_df.columns if 'percentage' in col.lower()]
            for col in percentage_cols:
                sectors_df[col] = pd.to_numeric(sectors_df[col], errors='coerce')
            
            # Identify and convert emission columns
            emission_cols = [col for col in sectors_df.columns 
                           if any(term in col for term in ['CO₂', 'CO2', 'tonnes'])]
            for col in emission_cols:
                sectors_df[col] = pd.to_numeric(sectors_df[col].str.replace(',', ''), errors='coerce')
            
            return sectors_df
            
        except Exception as e:
            print(f"Error in preprocess_ghg_sectors: {str(e)}")
            return pd.DataFrame()

    def combine_emissions_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Combine all emissions data with improved error handling."""
        emissions_dfs = []
        
        # Process each emissions dataset
        for filename, df in data_dict.items():
            try:
                # Extract emission type from filename
                emission_type = filename.split('_')[0]
                if emission_type in ['CO2', 'CH4', 'N2O', 'NOx', 'SO2', 'GHG']:
                    cleaned_df = self.clean_emissions_data(df, emission_type)
                    if not cleaned_df.empty:
                        emissions_dfs.append(cleaned_df)
                        print(f"Successfully processed {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Combine all emissions data
        if emissions_dfs:
            combined_emissions = pd.concat(emissions_dfs, ignore_index=True)
        else:
            combined_emissions = pd.DataFrame()
        
        # Process sector data separately
        sectors_df = pd.DataFrame()
        if 'GHG_Emissions_by_Sector' in data_dict:
            sectors_df = self.preprocess_ghg_sectors(data_dict['GHG_Emissions_by_Sector'])
        
        return combined_emissions, sectors_df

    def create_analysis_ready_dataset(self, 
                                    combined_emissions: pd.DataFrame, 
                                    sectors_df: pd.DataFrame) -> pd.DataFrame:
        """Create final analysis-ready dataset with improved error handling."""
        try:
            if combined_emissions.empty:
                print("No emissions data to process")
                return pd.DataFrame()
            
            # Melt the combined emissions data to long format
            id_vars = ['Country ID', 'Country', 'emission_type']
            value_vars = [col for col in combined_emissions.columns 
                         if col not in id_vars and col.isdigit()]
            
            melted_emissions = pd.melt(
                combined_emissions,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name='year',
                value_name='emission_value'
            )
            
            # Convert year to integer and sort
            melted_emissions['year'] = pd.to_numeric(melted_emissions['year'])
            melted_emissions = melted_emissions.sort_values(['Country', 'emission_type', 'year'])
            
            # Add sector information if available
            if not sectors_df.empty:
                sector_cols = [col for col in sectors_df.columns if 'percentage' in col.lower()]
                melted_emissions = melted_emissions.merge(
                    sectors_df[['Country ID', 'Latest Year Available'] + sector_cols],
                    on='Country ID',
                    how='left'
                )
            
            return melted_emissions
            
        except Exception as e:
            print(f"Error in create_analysis_ready_dataset: {str(e)}")
            return pd.DataFrame()