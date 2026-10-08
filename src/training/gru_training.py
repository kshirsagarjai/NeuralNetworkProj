"""Gru model training loop.

Contains:
- train_gru_model: A function to train GRU model with the specified parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models.gru_classifier import GRUClassifier


def train_gru_model(
    model: GRUClassifier,
    train_seqs: torch.Tensor,
    train_labels: torch.Tensor,
    val_seqs: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
    epochs: int = 100,
    learning_rate: float = 0.01,
    l2_weight: float = 0,
    batch_size: int | None = None,
    epoch_patience: int = 3,
    use_writer: bool = False,
    log_dir: str = "runs/experimen1",
) -> tuple[GRUClassifier, float | None, int | None]:
    """Train a model with the specified parameters.

    Args:
        model (GRUClassifier): The model that needs to be trained.
        train_seqs (torch.Tensor): The input training sequences of shape (n_sequences, seq_len, input_size)
        train_labels (torch.Tensor): The corresponding training labels of shape (n_sequences, 1)
        val_seqs (torch.Tensor | None, optional): The input validation sequences of shape (n_sequences, seq_len,
            input_size). If it is not supplied, there is no evaluation and no early stopping. Defaults to None.
        val_labels (torch.Tensor | None, optional): The corresponding training labels of shape (n_sequences, 1).
            If it is not supplied, there is no evaluation and no early stopping. Defaults to None.
        epochs (int, optional): The maximum number of epochs before the model is returned. Defaults to 100.
        learning_rate (float, optional): The learning rate for SGD. Defaults to 0.01.
        l2_weight (float, optional): The multiplier of the l2 term in SGD. Small amounts might help with regularization.
        batch_size (int | None, optional): The number of sequences in each batch. If no batch size is provided, all
            sequences are placed in one batch. Defaults to None.
        epoch_patience (int, optional): The maximum number of bad epochs (no improvement in validation loss), before
            the training loop is stopped. Defaults to 3.
        use_writer (bool, optional): Whether to log the train and validation losses for Tensorboard. Defaults to False.
        log_dir (str, optional): The name under which the losses need to be recorded. Defaults to "runs/experiment1".

    Raises:
        ValueError: If the number of training samples does not match the number of training labels.
        ValueError: If the number of validation samples does not match the number of validation labels.

    Returns:
        tuple[GRUClassifier, float |None, int | None]: (best_model, best_loss, best_epoch) The model with the lowest
            validation loss under the provided parameters. And the lowest validation loss if the validation set is
            provided, otherwise None. The epoch in which the best validation loss was achieved, returns None without
            validation set.
    """
    device = model.device
    validating = False
    best_val_loss = None
    best_epoch = None

    # Training data
    if train_seqs.size(0) != train_labels.size(0):
        raise ValueError(
            f"The number of training sequences {train_seqs.size(0)} does not match"
            f" the number of training labels {train_labels.size(0)}."
        )
    train_seqs = train_seqs.to(device)
    train_labels = train_labels.to(device)

    # Validation data
    if val_seqs is not None and val_labels is not None:
        if val_seqs.size(0) != val_labels.size(0):
            raise ValueError(
                f"The number of validation sequences {val_seqs.size(0)} does not match"
                f" the number of validation labels {val_labels.size(0)}."
            )
        val_seqs = val_seqs.to(device)
        val_labels = val_labels.to(device)

        # Variables for early stopping
        best_val_loss = float("inf")
        bad_epochs = 0
        validating = True

    loss_func = nn.CrossEntropyLoss()
    # l2_weight decay helps keep weights smaller, potentially improving generalization
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=l2_weight)

    n_seqs = train_seqs.size(0)

    # The writer for logging losses in Tensorboard
    writer = SummaryWriter(log_dir=log_dir) if use_writer else None

    # If no batch_size is specified, place all the data in one batch
    if batch_size is None:
        batch_size = n_seqs

    # Train for n epochs or when the loss starts increasing (early stopping)
    for epoch in range(int(epochs)):
        # Make sure it is not in evaluation mode
        model.train()
        epoch_loss = 0.0

        # === TRAINING ===
        # Update the gradients for every batch
        for batch_idx in range(0, n_seqs, batch_size):
            batch_train_seqs = train_seqs[
                batch_idx : batch_idx + batch_size
            ]  # shape: (batch_size, seq_len, input_size)
            batch_train_labels = train_labels[batch_idx : batch_idx + batch_size]  # shape: (batch_size)

            # First reset gradients
            optimizer.zero_grad()

            # model(x) automatically calls forward func and makes sure gradients are captured
            batch_logits = model(batch_train_seqs)

            batch_loss = loss_func(batch_logits, batch_train_labels)
            batch_loss.backward()
            optimizer.step()

            # Collect the weighted losses
            epoch_loss += batch_loss * batch_train_seqs.size(0)

        avg_epoch_loss = epoch_loss / n_seqs

        if writer:
            writer.add_scalar("Train Loss", avg_epoch_loss, epoch + 1)

        # === VALIDATION ===
        if validating:
            # We dont want to update the gradients
            model.eval()
            with torch.no_grad():
                val_logits = model(val_seqs)
                val_loss = loss_func(val_logits, val_labels)

            if writer:
                writer.add_scalar("Val Loss", val_loss, epoch + 1)

            # Use early stopping with some patience
            if val_loss < best_val_loss:
                # Store the performance and the new best epoch
                best_val_loss = val_loss
                best_epoch = epoch + 1

                # Store the best model so that it can be rebuild
                best_model_state = model.state_dict()
                # Reset patience counter since a new best model has been found
                bad_epochs = 0
            else:
                bad_epochs += 1
                # Stop if there are too many epochs without an decrease in val loss
                if bad_epochs >= epoch_patience:
                    print(f"Stopped at Epoch {epoch + 1}, best val_loss:{best_val_loss} at epoch {best_epoch}.")
                    model.load_state_dict(best_model_state)

                    # Return the model with the best val loss
                    return model, best_val_loss, best_epoch

    if validating:
        model.load_state_dict(best_model_state)
        print(f"Stopped at the final epoch: {epochs}, best val_loss:{best_val_loss} at epoch {best_epoch}.")

    if writer:
        writer.close()

    # When this return is called, either the model was in train only modus or it was still improving
    return model, best_val_loss, best_epoch
