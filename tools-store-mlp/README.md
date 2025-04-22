
## Files Description

### `datasets/code/create-dataset.py`

This script generates a synthetic dataset for tool sales. It simulates daily sales data for different tools over a year.

**Key Features:**

*   **Products:** Simulates sales for 'Hammer', 'Screwdriver', 'Pliers', 'Saw', and 'Drill'.
*   **Date Range:** Generates data for 365 days, starting from January 1, 2022.
*   **Features:**
    *   `date`: The date of the sale.
    *   `product`: The name of the tool.
    *   `week_day`: The day of the week (0-6).
    *   `month`: The month of the year (1-12).
    *   `holiday`: Indicates if the day is a holiday (1 for New Year's Day, 0 otherwise).
    *   `promotion`: Indicates if there's a promotion (1 or 0).
    *   `price`: The price of the tool.
    *   `previous_sales`: The demand of the tool in the previous day.
    *   `demand`: The number of units sold.
*   **Demand Simulation:** The demand is influenced by the day of the week, the month, promotions, and random variation.
*   **Output:** Saves the generated data to `datasets/csv/tools.csv`.

### `mlp_model.py`

This file defines the architecture of the MLP model using PyTorch.

**Model Details:**

*   **Class:** `ToolsDemandPredictor`
*   **Layers:**
    *   Input layer: Receives the input features.
    *   Hidden layer 1: `nn.Linear(input_size, hidden_size1)` followed by `nn.ReLU()`.
    *   Hidden layer 2: `nn.Linear(hidden_size1, hidden_size2)` followed by `nn.ReLU()`.
    *   Output layer: `nn.Linear(hidden_size2, output_size)` (predicts the demand).
*   **Activation:** ReLU (Rectified Linear Unit) is used for hidden layers.

### `mlp_pre.py`

This script handles the preprocessing of the data before it's fed into the model.

**Preprocessing Steps:**

1.  **Load Data:** Reads the `tools.csv` file into a Pandas DataFrame.
2.  **One-Hot Encoding:** Converts categorical features (`product`, `week_day`, `month`) into numerical representations using one-hot encoding.
3.  **Numerical Feature Scaling:** Applies a logarithmic transformation to `price` and `previous_sales` to handle potential skewness and large values.
4.  **Data Splitting:** Splits the data into training and validation sets (80% training, 20% validation).
5.  **Data Conversion:** Converts the data into NumPy arrays and then into PyTorch tensors.
6.  **Output:** Returns the training and validation sets (`X_train`, `X_val`, `y_train`, `y_val`) and the number of input features (`n_features`).

### `mlp_processing.py`

This is the main script that orchestrates the training and evaluation of the MLP model.

**Workflow:**

1.  **Preprocessing:** Calls `preprocessing()` from `mlp_pre.py` to load and preprocess the data.
2.  **Model Instantiation:** Creates an instance of the `ToolsDemandPredictor` model from `mlp_model.py`.
3.  **Loss Function and Optimizer:** Defines the Mean Squared Error (MSE) loss function and the Adam optimizer.
4.  **DataLoaders:** Creates PyTorch `DataLoader` objects for the training and validation sets.
5.  **Training Loop:**
    *   Iterates through the training data for a specified number of epochs.
    *   Calculates the training loss for each epoch.
    *   Evaluates the model on the validation set after each epoch.
    *   Implements **Early Stopping**: Monitors the validation loss and stops training if it doesn't improve for a certain number of epochs (patience).
    *   Calculates and prints the **MAE**, **RMSE** and **R²** for each epoch.
    *   Saves the best model based on the lowest validation loss.
6.  **Final Evaluation:** After training, it loads the best model and evaluates it on the validation set, printing the final validation loss, MAE, RMSE and R².
7. **Metrics:**
    * **MAE (Mean Absolute Error):** Average absolute difference between predictions and actual values.
    * **RMSE (Root Mean Squared Error):** Square root of the average squared difference between predictions and actual values.
    * **R² (R-squared):** Proportion of the variance in the dependent variable that is predictable from the independent variables.

## How to Run the Code

1.  **Generate the Dataset:**
    ```bash
    cd tools-store-mlp/datasets/code
    python create-dataset.py
    ```
    This will create the `tools.csv` file in the `datasets/csv/` directory.

2.  **Train and Evaluate the Model:**
    ```bash
    cd ../../../
    python mlp_processing.py
    ```
    This will run the training and evaluation process, printing the results to the console.

## Dependencies

*   Python 3.x
*   Pandas
*   NumPy
*   Scikit-learn
*   PyTorch

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn torch
