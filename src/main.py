# main.py
import torch
import numpy as np
from pathlib import Path
from data.data_loader import EnvironmentalDataLoader
from data.preprocessor import EnvironmentalDataPreprocessor
from data.emission_preprocess import EmissionsPreprocessor
from models.trainer import ModelTrainer
from models.climate_net import EnvironmentalNet
from visualization.plotter import EnvironmentalPlotter
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchviz import make_dot
import matplotlib.pyplot as plt

def main():
    # Initialize components
    data_loader = EnvironmentalDataLoader()
    emission_preprocessor = EmissionsPreprocessor()
    data_preprocessor = EnvironmentalDataPreprocessor()
    
    # Load and process raw data
    try:
        print("\nLoading Air and Climate data...")
        raw_data = data_loader.load_category('Air and Climate')
        
        # Process emissions data
        combined_emissions, sectors_data = emission_preprocessor.combine_emissions_data(raw_data)
        analysis_df = emission_preprocessor.create_analysis_ready_dataset(combined_emissions, sectors_data)
        
        # Preprocess time series
        target_col = 'emission_value'
        processed_df = data_preprocessor.preprocess_timeseries(analysis_df, target_col)
        
    except Exception as e:
        print(f"Data processing failed: {str(e)}")
        return

    # Create sequences
    try:
        seq_length = data_preprocessor.config['model_params']['sequence_length']
        sequences, targets = data_preprocessor.create_sequences(
            processed_df[f'{target_col}_scaled'].values,
            seq_length
        )
        
        # Convert to tensors
        X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
        y = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        batch_size = data_preprocessor.config['model_params']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
    except Exception as e:
        print(f"Sequence creation failed: {str(e)}")
        return

    # Initialize model
    try:
        model_params = data_preprocessor.config['model_params']
        model = EnvironmentalNet(
            input_size=1,  # Using univariate time series
            hidden_size=model_params['hidden_size'],
            num_layers=model_params['num_layers']
        )
        
        # Visualize model architecture
        sample_input = torch.randn(1, seq_length, 1)
        make_dot(model(sample_input), params=dict(model.named_parameters())).render("model_architecture", format="png")
        print("Saved model architecture visualization as model_architecture.png")
        
    except Exception as e:
        print(f"Model initialization failed: {str(e)}")
        return

    # Train model
    try:
        trainer = ModelTrainer(
            model,
            learning_rate=model_params['learning_rate']
        )
        
        print("\nStarting training...")
        train_losses, val_losses = trainer.train(
            train_loader,
            val_loader,
            epochs=model_params['epochs']
        )
        
        # Plot training curves
        EnvironmentalPlotter.plot_loss_curves(train_losses, val_losses)
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return

    # Evaluate and visualize
    try:
        model.eval()
        with torch.no_grad():
            # Get sample predictions
            sample_x, sample_y = next(iter(val_loader))
            predictions = model(sample_x)
            
            # Inverse scaling
            scaler = data_preprocessor.scalers[target_col]
            actual = scaler.inverse_transform(sample_y.numpy().reshape(-1, 1))
            predicted = scaler.inverse_transform(predictions.numpy().reshape(-1, 1))
            
            # Plot predictions
            EnvironmentalPlotter.plot_time_series(
                actual.flatten(),
                predicted.flatten(),
                title="Emission Predictions vs Actual Values"
            )
            
            # Save model
            torch.save(model.state_dict(), "environmental_net.pth")
            print("\nSaved trained model as environmental_net.pth")
            
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return

if __name__ == "__main__":
    main()