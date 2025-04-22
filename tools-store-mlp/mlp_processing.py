from mlp_model import ToolsDemandPredictor
from mlp_pre import preprocessing
from sklearn.metrics import r2_score  # Importe a função r2_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def calculate_mae_rmse(predictions, targets):
    mae = torch.mean(torch.abs(predictions - targets))
    rmse = torch.sqrt(torch.mean((predictions - targets)**2))
    return mae, rmse

def main():
    # 1. Pré-processamento dos dados
    X_train, X_val, y_train, y_val, n_features = preprocessing('datasets/csv/tools.csv')

    # 2. Definir as dimensões do modelo
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = 1

    # 3. Instanciar o modelo
    model = ToolsDemandPredictor(n_features, hidden_size1, hidden_size2, output_size)
    print("\nInstantiated model:")
    print(model)

    # 4. Definir função de perda e otimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nLoss function and optimizer defined.")

    # 5. Criar DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)

    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    print("\nStarting training with Early Stopping...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        all_val_predictions = []
        all_val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                all_val_predictions.append(outputs)
                all_val_targets.append(targets)
        epoch_val_loss = val_loss / len(val_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_val_loss:.4f}')

        all_val_predictions_tensor = torch.cat(all_val_predictions, dim=0)
        all_val_targets_tensor = torch.cat(all_val_targets, dim=0)
        mae, rmse = calculate_mae_rmse(all_val_predictions_tensor, all_val_targets_tensor)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation MAE: {mae:.4f}, Validation RMSE: {rmse:.4f}')

        r2 = r2_score(all_val_targets_tensor.numpy(), all_val_predictions_tensor.numpy())
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation R²: {r2:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # Salva os pesos do melhor modelo
        else:
            epochs_without_improvement += 1
            print(f'Epoch [{epoch+1}/{num_epochs}], No improvement in validation loss for {epochs_without_improvement} epochs.')
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}.')
                break

    # Carrega os pesos do melhor modelo encontrado
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print('\nLoaded best model weights based on validation loss.')

    print("\nTraining completed.")

    # Avaliação final no conjunto de validação com o melhor modelo
    model.eval()
    final_val_loss = 0.0
    final_all_val_predictions = []
    final_all_val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            final_val_loss += loss.item() * inputs.size(0)
            final_all_val_predictions.append(outputs)
            final_all_val_targets.append(targets)
    final_epoch_val_loss = final_val_loss / len(val_dataset)
    final_all_val_predictions_tensor = torch.cat(final_all_val_predictions, dim=0)
    final_all_val_targets_tensor = torch.cat(final_all_val_targets, dim=0)
    final_mae, final_rmse = calculate_mae_rmse(final_all_val_predictions_tensor, final_all_val_targets_tensor)
    final_r2 = r2_score(final_all_val_targets_tensor.numpy(), final_all_val_predictions_tensor.numpy())
    print(f'\nFinal Validation Loss: {final_epoch_val_loss:.4f}')
    print(f'Final Validation MAE: {final_mae:.4f}'),
    print(f'Final Validation RMSE: {final_rmse:.4f}')
    print(f'Final Validation R²: {final_r2:.4f}')

if __name__ == "__main__":
    main()