"""Gru model training loop.

Contains:
- hyperparameter_search_gru: A function that searches the parameter combination that results in the lowest val loss.
- k_fold_gru: A function to train and evaluate a given set of parameters using K-fold crossvalidation
"""

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from tqdm import tqdm

from src.models.gru_classifier import GRUClassifier
from src.training.gru_training import train_gru_model


def hyperparameter_search_gru(
    param_grid: dict[str, list[int | float | bool]],
    train_seqs: torch.Tensor,
    train_labels: torch.Tensor,
    n_folds: int = 5,
    device: torch.device | None = None,
) -> dict[str, int | float | bool] | None:
    """Find the hyperparameters that result in the lowest validation loss using K-fold cross validation.

    Args:
        param_grid (dict[str, list[int | float | bool]]): A dictionary containing a range of values for each
            hyperparameter of the training loop and the model. Must include: hidden_size, bidirectional, epochs,
            learning_rate, batch_size and epoch_patience.
        train_seqs (torch.Tensor): Shape: (n_sequences, seq_len, input_size) The sequences that need to be split into
            training and validation sets to train the model.
        train_labels (torch.Tensor): Shape: (n_sequences,) The corresponding labels that need to be split into training
            and validation sets to train the model.
        n_folds (int, optional): The number of folds the data needs to be split into. Defaults to 5.
        device (torch.device | None, optional): The device to use for the tensor operations. If no device is
            specified, "cpu" is used. Defaults to None.

    Returns:
        dict[str, int | float | bool]: The combination of parameters that resulted in the lowest valuation loss.
    """
    best_val_loss = float("inf")
    best_params = None

    param_grid = list(ParameterGrid(param_grid))

    for params in tqdm(param_grid, desc="Evaluating parameter combinations"):
        val_loss, avg_epochs = k_fold_gru(params, train_seqs, train_labels, n_folds, device)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            # Make a copy of the parameter combination
            best_params = dict(params)
            # Add the average epochs that it took to reach this eval score
            best_params["epochs"] = avg_epochs

    return best_params


def k_fold_gru(
    params: dict[str, int | float | bool],
    train_seqs: torch.Tensor,
    train_labels: torch.Tensor,
    n_folds: int = 5,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Evaluate the model with the provided parameters using K-fold cross-validation.

    Args:
        params (dict[str, int | float | bool]): A dictionary containing the hyperparameters of the training loop and
            the model. Must include: hidden_size, bidirectional, epochs, learning_rate, batch_size and epoch_patience
        train_seqs (torch.Tensor): Shape: (n_sequences, seq_len, input_size) The sequences that need to be split into
            training and validation sets to train the model.
        train_labels (torch.Tensor): Shape: (n_sequences,) The corresponding labels that need to be split into training
            and validation sets to train the model.
        n_folds (int, optional): The number of folds the data needs to be split into. Defaults to 5.
        device (torch.device | None, optional): The device to use for the tensor operations. If no device is
            specified, "cpu" is used. Defaults to None.

    Returns:
        tuple[float, float]: (avg_val_loss, avg_epochs) The average evaluation loss and the average number of epochs to
            reach the best evaluation loss.
    """
    device = device if device else torch.device("cpu")

    # Store the number of MFCC channels
    input_size = train_seqs.size(2)

    # Place the parameters in the run name for easier tracking in Tensorboard
    parameters_str = ".".join(f"{key}={value}" for key, value in params.items())
    print(parameters_str)

    # Splits the labels while keeping equal distribution in each fold
    data_splitter = StratifiedKFold(n_splits=n_folds)

    # The splitter can only use numpy labels, it does not need the actual sequences
    numpy_labels = train_labels.detach().cpu().numpy()
    n_seqs = len(numpy_labels)
    n_unique_labels = len(np.unique(numpy_labels))

    avg_val_loss = 0
    avg_epochs = 0

    for fold, (train_idx, val_idx) in enumerate(data_splitter.split(np.zeros(n_seqs), numpy_labels)):
        # Log each indiviual fold to check if they behave as expected
        log_dir = f"runs/fold_{fold+1}/{parameters_str}"

        # Split the training set into the appropriate train and validation sets
        train_seqs_fold = train_seqs[train_idx]
        train_labels_fold = train_labels[train_idx]
        val_seqs_fold = train_seqs[val_idx]
        val_labels_fold = train_labels[val_idx]

        gru_model = GRUClassifier(
            input_size, params["hidden_size"], n_unique_labels, params["bidirectional"], params["dropout_prob"], device
        )

        _, val_loss, best_epoch = train_gru_model(
            model=gru_model,
            train_seqs=train_seqs_fold,
            train_labels=train_labels_fold,
            val_seqs=val_seqs_fold,
            val_labels=val_labels_fold,
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            l2_weight=params["l2_weight"],
            batch_size=params["batch_size"],
            epoch_patience=params["epoch_patience"],
            use_writer=True,
            log_dir=log_dir,
        )
        if val_loss is not None:
            avg_val_loss += float(val_loss)
        if best_epoch is not None:
            avg_epochs += int(best_epoch)

    return avg_val_loss / n_folds, avg_epochs / n_folds
