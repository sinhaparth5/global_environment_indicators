import matplotlib.pyplot as plt
import seaborn as sns

class EnvironmentalPlotter:
    @staticmethod
    def plot_time_series(actual, predicted, title="Modal Predictions vs Actual Values"):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual', color='blue', alpha=0.7)
        plt.plot(predicted, label='Predicted', color='red', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_loss_curves(train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.title('Training and Validation Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_category_comparison(data, category_column, value_column):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=data, x=category_column, y=value_column)
        plt.xticks(rotation=45)
        plt.title(f'{value_column} by {category_column}')
        plt.tight_layout()
        plt.show()