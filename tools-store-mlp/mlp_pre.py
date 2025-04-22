import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def preprocessing(csv_path):
    # 1. Load the dataset from the CSV file
    df = pd.read_csv(csv_path)

    # 2. Pre-process the Features
    # One-Hot Encode Categorical Variables
    df = pd.get_dummies(df, columns=['product', 'week_day', 'month'], drop_first=False)

    # Scale Numerical Features
    df['price'] = df['price'].apply(lambda x: np.log(x + 1))  # Evita valores negativos ou zero
    df['previous_sales'] = df['previous_sales'].apply(lambda x: np.log(x + 1))

    # 3. Prepare Data for PyTorch
    # Separate features (X) and target (y)
    X = df.drop(['date', 'demand'], axis=1)
    y = df['demand']

    # Convert to NumPy arrays
    X_np = X.values.astype('float32')
    y_np = y.values.astype('float32').reshape(-1, 1)

    # 4. Split Data into Training and Validation Sets
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )

    # 5. Convert to PyTorch Tensors
    X_train = torch.tensor(X_train_np)
    X_val = torch.tensor(X_val_np)
    y_train = torch.tensor(y_train_np)
    y_val = torch.tensor(y_val_np)

    # Print the shapes of the tensors for verification
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)

    # The number of input features for our network should be 28
    n_features = X_train.shape[1]
    print("\nNumber of input features:", n_features)

    return X_train, X_val, y_train, y_val, n_features