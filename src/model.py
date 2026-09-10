"""
The PyTorch regression model.

A two-hidden-layer MLP with ReLU activations, wrapped so that it exposes the same
`fit` / `predict` interface as the scikit-learn estimators it is compared against.
Training uses Adam, MSE loss and early stopping on validation loss.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SimpleNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, learning_rate, epochs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate        
        self.epochs = epochs
     
        # Define the structure of the neural network
        self.mlp = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),                
                nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self,x):
        x = self.mlp(x)
        return x
    

    def fit(self, X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, verbose=False):

        # Define the model
        #self.model = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

        X_train_tensor = torch.from_numpy(X_train_scaled).float()
        y_train_tensor = torch.from_numpy(y_train_scaled).float()

        X_valid_tensor = torch.from_numpy(X_valid_scaled).float()
        y_valid_tensor = torch.from_numpy(y_valid_scaled).float()

        # Create a DataLoader for the training data    
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        
        # Create a DataLoader for the validation data
        val_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        best_val_loss = float('inf')    # Initialize to infinity
        patience = 150                  # Number of epochs to wait before stopping
        counter = 0      

        # Create the history of the loss
        train_loss_history = []
        val_loss_history = []

        # Loop over the epochs with tqdm
        for epoch in tqdm(range(self.epochs), desc="Training Progress", unit="epoch"):
        
            # Set the model in training mode
            self.train()

            for X_batch, y_batch in train_loader:

                # Prediction
                train_pred = self(X_batch)

                # Loss calculation
                loss = self.loss_function(train_pred, y_batch)

                # Calculate the gradient
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                # Reset the gradient
                self.optimizer.zero_grad()        

            # Add the loss value to the history
            train_loss_history.append(loss.item())

            # Set the model in evaluation mode
            self.eval()
            with torch.no_grad():   

                # Prediction and loss calculation
                for X_batch, y_batch in val_loader:
                    valid_pred = self(X_batch)
                    val_loss = self.loss_function(valid_pred, y_batch) 

                # Add the loss value to the history
                val_loss_history.append(val_loss.item())


            # if (epoch + 1) % 100 == 0:            
            #    print(f'TRAINING - Epoch: {epoch+1} | Training Loss: {loss.item():.7f} | Validation Loss: {val_loss.item():.7f} ')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if verbose:
            # Visualizing the loss history
            plt.figure()
            plt.plot(train_loss_history, label="Training Loss")
            plt.plot(val_loss_history, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()  
      

    def predict(self, X):

        self.eval()
        X_tensor = torch.from_numpy(X).float()

        with torch.inference_mode():
            # Prediction
            y_pred =self(X_tensor).squeeze()

        return y_pred.detach().numpy()
