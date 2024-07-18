# TensorBoard for logging training metrics and visualizations
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb


# Model training loop
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    batch_size,
    num_epochs,
    optimizer,
    loss_fn,
    train_acc_metric,
    val_acc_metric,
):
    """
    Trains a model using the given data and hyperparameters.

    Args:
        model (keras.Model): The model to be trained.
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training set.
        val_dataloader (torch.utils.data.DataLoader): The data loader for the validation set.
        batch_size (int): The batch size for training and validation.
        num_epochs (int): The number of epochs to train the model.
        optimizer (keras.optimizers.Optimizer): The optimizer used for updating the model's parameters.
        loss_fn (keras.losses.Loss): The loss function used for computing the training loss.
        train_acc_metric (keras.metrics.Metric): The training accuracy metric.
        val_acc_metric (keras.metrics.Metric): The validation accuracy metric.

    Returns:
        None
    """

    # Create a SummaryWriter for logging TensorBoard events
    train_writer = SummaryWriter(log_dir="logs/train")
    val_writer = SummaryWriter(log_dir="logs/val")

    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        # Print the current epoch
        print(f"\nStart of epoch {epoch}")

        for step, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(
                non_blocking=True
            )

            # Forward pass of the model
            logits = model(inputs)
            loss = loss_fn(targets, logits)

            # Backward pass for the gradients
            model.zero_grad()
            trainable_weights = [v for v in model.trainable_weights]

            # Call torch.Tensor.backward() on the loss to compute gradients for the weights.
            loss.backward()
            gradients = [v.value.grad for v in trainable_weights]

            # Update weights
            with torch.no_grad():
                optimizer.apply(gradients, trainable_weights)

            # Update training metrics
            train_acc_metric.update_state(targets, logits)

            # Log training loss to TensorBoard
            train_writer.add_scalar(
                "Loss", loss.item(), epoch * len(train_dataloader) + step
            )

            # Log training loss to wandb
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

            # Log every 100 batches to track progress of training
            if step % 100 == 0:
                print(
                    f"Training loss (for 1 batch) at step {step}: {loss.cpu().detach().numpy():.4f}"
                )
                print(f"Seen so far: {(step + 1) * batch_size} samples")

        # Display training metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print(f"Training acc over epoch: {float(train_acc):.4f}")

        # Log training accuracy to TensorBoard
        train_writer.add_scalar("Accuracy", train_acc.item(), epoch)

         # Log training accuracy to wandb
        wandb.log({"train_accuracy": train_acc.item(), "epoch": epoch})

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_state()

        # Run a validation loop at the end of each epoch
        for x_batch_val, y_batch_val in val_dataloader:
            # validation set logits
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)

        # Validation metrics at the end of each epoch
        val_acc = val_acc_metric.result()

        # Log validation accuracy to TensorBoard
        val_writer.add_scalar("Accuracy", val_acc, epoch)

         # Log validation accuracy to wandb
        wandb.log({"val_accuracy": val_acc.item(), "epoch": epoch})

        # Reset validation metrics at the end of each epoch
        val_acc_metric.reset_state()

        # Print validation accuracy
        print(f"Validation acc: {float(val_acc):.4f}")

    # Close the SummaryWriter
    train_writer.close()
    val_writer.close()
