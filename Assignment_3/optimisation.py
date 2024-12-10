# hyperparameter_optimization.py
import itertools
from training_evaluation import train_and_evaluate
import torch.nn as nn
import torch.optim as optim

class HyperparameterOptimizer:
    def __init__(self, model_class, train_loader, val_loader, rnn_type):
        
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rnn_type = rnn_type
        self.best_params = None
        self.best_metrics = None
        self.best_model = None
        self.results = []
        
    def _get_optimizer(self, optimizer_name, model_parameters, lr):
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(model_parameters, lr=lr, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            return optim.RMSprop(model_parameters, lr=lr)
        else:
            raise ValueError(f"Wrong optimizer: {optimizer_name}")
        
    def optimize(self, param_grid, num_epochs=50, patience=5):
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        best_val_loss = float('inf')
        total_combinations = len(combinations)
        
        print(f'\noptimize {self.rnn_type} model')
        print(f'Number of combinations: {total_combinations}')
        
        for i, combo in enumerate(combinations, 1):
            print(f'\nTesting combination {i}/{total_combinations}')
            
            # Create a dictionary with actual parameters
            current_params = dict(zip(keys, combo))
            
            # Extract optimizer parameter
            optimizer_name = current_params.pop('optimizer', 'adam')
            learning_rate = current_params.pop('learning_rate', 0.001)
            
            # Add rnn-type
            current_params['rnn_type'] = self.rnn_type
            
            # Initialise model with actual hyperparameters
            model = self.model_class(**current_params)
            
            optimizer = self._get_optimizer(optimizer_name, model.parameters(), learning_rate)
            criterion = nn.MSELoss()
            
            train_losses, val_losses, mse_values, mae_values, rmse_values, r2_values = train_and_evaluate(
                model, 
                self.train_loader, 
                self.val_loader, 
                criterion, 
                optimizer,
                num_epochs=num_epochs,
                patience=patience
            )
            
            best_epoch_idx = val_losses.index(min(val_losses))
            current_metrics = {
                'val_loss': val_losses[best_epoch_idx],
                'mse': mse_values[best_epoch_idx],
                'mae': mae_values[best_epoch_idx],
                'rmse': rmse_values[best_epoch_idx],
                'r2': r2_values[best_epoch_idx]
            }
            
            full_params = {
                **current_params,
                'optimizer': optimizer_name,
                'learning_rate': learning_rate
            }
            
            self.results.append({
                'params': full_params,
                'metrics': current_metrics
            })
            
            if current_metrics['val_loss'] < best_val_loss:
                best_val_loss = current_metrics['val_loss']
                self.best_params = full_params
                self.best_metrics = current_metrics
                self.best_model = model.state_dict().copy()
                
            print(f"Parameters: {full_params}")
            print(f"Metrics: {current_metrics}")
            
    def get_best_results(self):
        return {
            'rnn_type': self.rnn_type,
            'best_parameters': self.best_params,
            'best_metrics': self.best_metrics,
            'all_results': self.results
        }