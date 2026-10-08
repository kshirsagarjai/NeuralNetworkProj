"""Train and store a model using K-fold hyperparameter search."""

from pathlib import Path

import torch

from src.models.gru_classifier import GRUClassifier
from src.training.gru_training import train_gru_model
from src.training.k_fold_hyperparameter_search import hyperparameter_search_gru
from src.utils import file_to_sequences, make_label_array

# Resolve makes current path absolute and parent goes back one step to NN-SEMESTER-PROJECT
ROOT = Path(__file__).resolve().parent
MODEL_FILE_NAME = "gru_final.pth"
train_file_path = ROOT / "data" / "ae.train"
save_dir = ROOT / "saved_models"

N_FOLDS = 5


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training GRU on GPU...")
    else:
        device = torch.device("cpu")
        print("No GPU detected or using non-cuda torch version, training on CPU...")

    # === HYPERPARAMETER SEARCH ===
    # Input must be tensors (n_sequences, seq_len, n_channels)
    # Also, the sequences must be padded to the same length
    train_seqs = torch.from_numpy(file_to_sequences(train_file_path, pad_sequences=True))
    # Input must be tensor (n_sequences) (30 times each label)
    label_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30]
    train_labels = torch.from_numpy(make_label_array(label_counts))

    # Note that epoch patience will not be used for final model
    # it is only there to help find the correct epoch number using early stopping
    param_grid = {
        "hidden_size": [32, 64, 128],
        "bidirectional": [True],
        "dropout_prob": [0.0, 0.2, 0.4],
        "epochs": [3000],
        "learning_rate": [1e-2, 1e-3, 1e-4],
        "l2_weight": [0, 1e-4, 1e-3, 1e-2],
        "batch_size": [8, 16, 32, 64],
        "epoch_patience": [5],
    }

    best_params = hyperparameter_search_gru(param_grid, train_seqs, train_labels, N_FOLDS, device)
    print("Best found hyperparameters:", best_params)

    # === FINAL TRAINING ===
    # Only initialze the model with the best_parameters
    input_size = train_seqs.size(2)
    n_labels = len(torch.unique(train_labels))

    gru_model = GRUClassifier(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        n_classes=n_labels,
        bidirectional=best_params["bidirectional"],
        dropout_prob=best_params["dropout_prob"],
        device=device,
    ).to(device)

    # Train the final model on all training sequences without validation
    # Note that it will only track the training loss
    best_model, _, _ = train_gru_model(
        gru_model,
        train_seqs,
        train_labels,
        val_seqs=None,
        val_labels=None,
        epochs=best_params["epochs"],
        learning_rate=best_params["learning_rate"],
        l2_weight=best_params["l2_weight"],
        batch_size=best_params["batch_size"],
        epoch_patience=best_params["epoch_patience"],
        use_writer=True,
        log_dir="runs/final_training",
    )

    # Create the models folder if does not exist already and store it there
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / MODEL_FILE_NAME
    torch.save(best_model, save_path)
    print(f"Model saved to {save_path}.")
