import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class HyperspectralEvaluator:
    def __init__(self, model, test_data):
        self.model = model
        self.X_test, self.y_test = test_data
        
    def evaluate(self):
        # Make predictions
        with torch.no_grad():
            y_pred = self.model(self.X_test)
        
        # Calculate metrics using PyTorch on GPU
        mse = torch.mean((self.y_test - y_pred)**2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(self.y_test - y_pred))
        
        # Calculate R2 score
        ss_total = torch.sum((self.y_test - torch.mean(self.y_test))**2)
        ss_residual = torch.sum((self.y_test - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
        
        metrics = {
            'RMSE': rmse.item(),
            'R2': r2.item(),
            'MAE': mae.item()
        }
        
        # Plot results
        self._plot_predictions(y_pred)
        
        return metrics
        
    def _plot_predictions(self, y_pred):
        plt.figure(figsize=(12, 6))
        
        # Move tensors to CPU for plotting
        y_test_cpu = self.y_test.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        
        # Plot actual vs predicted for each parameter
        for i in range(y_test_cpu.shape[1]):
            plt.subplot(2, 2, i+1)
            plt.scatter(y_test_cpu[:, i], y_pred_cpu[:, i], alpha=0.5)
            plt.plot([min(y_test_cpu[:, i]), max(y_test_cpu[:, i])], 
                    [min(y_test_cpu[:, i]), max(y_test_cpu[:, i])], 
                    'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Parameter {i+1}')
            
        plt.tight_layout()
        plt.show()
