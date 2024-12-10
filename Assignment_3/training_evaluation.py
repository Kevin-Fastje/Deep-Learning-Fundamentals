import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=5):
    model.train()
    train_losses = []
    val_losses = []
    mse_values = []
    mae_values = []
    rmse_values = []
    r2_values = []
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            targets = targets.squeeze(1)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        valid_loss, mse, mae, rmse, r2 = evaluate(model, val_loader, criterion)
        val_losses.append(valid_loss)
        mse_values.append(mse)
        mae_values.append(mae)
        rmse_values.append(rmse)
        r2_values.append(r2)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {valid_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Early stopping logic
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping after epoch {epoch+1}')
            model.load_state_dict(best_model)
            break

    return train_losses, val_losses, mse_values, mae_values, rmse_values, r2_values


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for data, targets in val_loader:
            output = model(data)
            targets = targets.squeeze(1)
            loss = criterion(output, targets)
            val_loss += loss.item()
            
            targets_np = targets.numpy()
            predictions = output.numpy()
            
            targets_all.append(targets_np)
            predictions_all.append(predictions)

    val_loss /= len(val_loader)
    
    targets_all = np.concatenate(targets_all, axis=0)
    predictions_all = np.concatenate(predictions_all, axis=0)

    mse = mean_squared_error(targets_all, predictions_all)
    mae = mean_absolute_error(targets_all, predictions_all)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_all, predictions_all)

    return val_loss, mse, mae, rmse, r2



def plot_loss_curve(train_losses, val_losses):
    
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

def plot_metrics(mse_values, mae_values, rmse_values, r2_values):
    
    plt.plot(mse_values, label="MSE")
    plt.plot(mae_values, label="MAE")
    plt.plot(rmse_values, label="RMSE")
    plt.plot(r2_values, label="R²")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title("Metrics over Epochs")
    plt.show()
