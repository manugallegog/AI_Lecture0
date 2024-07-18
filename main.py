# Set the backend to PyTorch
import os
import wandb

# Import keras and torch
import keras
import torch

os.environ["KERAS_BACKEND"] = "torch"

# Import local modules
from src.dataset import get_dataset
from src.model import get_model
from src.train import train_model

# Add requirement for wandb core
wandb.require("core")

# Hyperparameters
batch_size = 32  # Batch size for training and validation datasets
num_epochs = 5  # Number of epochs for training the model
learning_rate = 1e-3 # Learning rate for the optimizer

# Initialize wandb
wandb.init(project="AI_Lecture0", config={
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate
})

# Get the training and validation datasets
train_dataset, val_dataset = get_dataset()

# Create DataLoaders for the Datasets using torch dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)


# Get the model
model = get_model()

# Define optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Define the training and validation metrics as CategoricalAccuracy
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()


if __name__ == "__main__":
    # Train the model
    train_model(
        model,
        train_dataloader,
        val_dataloader,
        batch_size,
        num_epochs,
        optimizer,
        loss_fn,
        train_acc_metric,
        val_acc_metric,
    )
    wandb.finish()
