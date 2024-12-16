import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
import time
from tqdm import tqdm
import itertools

from utils import *
from dataset import SiameseDataset, TripletDataset, initialize_datasets , initialize_datasets_triplet, load_triplets_data, load_data
from model import SiameseNetworkTriplet, SiameseNetwork

def train_model(model, train_loader, val_loader, test_loader=None, device='cpu', num_epochs=10, lr=0.0005,
                weight_decay=0.0005, patience=5, min_delta=0.1, save_path="model_logs"):
    """
    Train a Siamese model with pair-based validation.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data (pairs).
        val_loader: DataLoader for validation data (pairs).
        test_loader: Optional DataLoader for test data (pairs).
        device: Device to train on ('cpu' or 'cuda').
        num_epochs: Number of training epochs.
        lr: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization).
        patience: Number of epochs to wait for improvement before early stopping.
        min_delta: Minimum change in validation loss to qualify as an improvement.
        save_path: Path to save model and logs.

    Returns:
        epochs_run: Number of epochs run.
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("Run on", device)
    os.makedirs(save_path, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    accuracy_logs = []
    epochs_run = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        accuracy_logs.append({"epoch": epoch + 1, "val_accuracy": val_accuracy})

        log_epoch(epoch, num_epochs, train_loss, val_loss, {"val_accuracy": val_accuracy})

        epochs_run = epoch + 1

        # Early Stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    plot_losses(train_losses, val_losses, save_path)

    # Test evaluation
    if test_loader:
        test_accuracy = evaluate_classification_accuracy(model, test_loader, device)
        accuracy_logs.append({"test_accuracy": test_accuracy})
        log_test_results({"test_accuracy": test_accuracy})

    save_logs({"train_losses": train_losses, "val_losses": val_losses, "accuracy_logs": accuracy_logs}, save_path)
    return epochs_run, train_losses, val_losses


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for x1, x2, y in train_loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def train_triplet_model(
    model, train_loader, val_triplet_loader, val_pair_loader, 
    test_loader=None, device='cpu', num_epochs=10, lr=0.0005, 
    weight_decay=0.0005, patience=5, min_delta=0.1, save_path="model_logs", margin=1.0
):
    """
    Train a triplet model with pair-based validation.
    
    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data (triplets).
        val_triplet_loader: DataLoader for validation data (triplets).
        val_pair_loader: DataLoader for validation data (pairs).
        test_loader: Optional DataLoader for test data (pairs).
        device: Device to train on ('cpu' or 'cuda').
        num_epochs: Number of training epochs.
        lr: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization).
        patience: Number of epochs to wait for improvement before early stopping.
        min_delta: Minimum change in validation loss to qualify as an improvement.
        save_path: Path to save model and logs.
        margin: Margin for the triplet loss.

    Returns:
        epoch_run: Number of epochs run.
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
    """
    # Initialize loss and optimizer
    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Run on", device)
    set_seed(2024)
    os.makedirs(save_path, exist_ok=True)

    # Initialize tracking variables
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    accuracy_logs = []

    for epoch in range(num_epochs):
        train_loss = train_epoch_triplet(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss, val_metrics = validate_model_triplet(model, val_pair_loader, val_triplet_loader, device)
        val_losses.append(val_loss)
        accuracy_logs.append(val_metrics)

        log_epoch_triplet(epoch, num_epochs, train_loss, val_loss, val_metrics)

        if early_stopping(val_loss, best_val_loss, min_delta, patience_counter, patience):
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_triplet_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save plots and logs
    plot_losses(train_losses, val_losses, save_path)
    if test_loader:
        test_metrics = evaluate_test_model(model, test_loader, device, triplet=True)
        accuracy_logs.append(test_metrics)
        log_test_results(test_metrics)

    save_logs({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "accuracy_logs": accuracy_logs,
    }, save_path)

    return len(train_losses), train_losses, val_losses

def train_epoch_triplet(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for anchor, positive, negative in train_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

# ------------- Grid Search Siemse ------------- #

def initialize_model(network_size, use_batchnorm, use_dropout, dropout, device):
    """Initialize and return the model."""
    return SiameseNetwork(
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
        network_size=network_size,
        dropout_rate=dropout
    ).to(device)
    
# ------------- Grid Search Siemse ------------- #

def grid_search_siemse(datast_path, device, param_grid, experiment_dir= './grid_search', num_epochs=20):
    """
    Perform grid search over hyperparameters to train and evaluate the model.

    Args:
        train_data (dict): Training dataset components (x_first, x_second, labels).
        val_data (dict): Validation dataset components (x_first, x_second, labels).
        test_data (dict): Test dataset components (x_first, x_second, labels).
        device (str): Device to train the model on (e.g., 'cpu' or 'cuda').
        param_grid (dict): Dictionary defining hyperparameter options.
        num_epochs (int): Number of training epochs.

    Returns:
        dict: Results containing performance metrics for each configuration.
    """
    results = {}
    train_data, val_data, test_data = load_data(datast_path)
    os.makedirs(experiment_dir, exist_ok=True)

    # Calculate the total number of configurations
    total_configs = len(list(itertools.product(*param_grid.values())))

    # Add tqdm progress bar
    with tqdm(total=total_configs, desc="Grid Search Progress") as pbar:
        for config in itertools.product(*param_grid.values()):
            # Extract configuration
            config_dict = dict(zip(param_grid.keys(), config))
            dropout = config_dict['dropout']
            weight_decay = config_dict['weight_decay']
            lr = config_dict['lr']
            batch_size = config_dict['batch_size']
            network_size = config_dict['network_size']
            use_batchnorm = config_dict['use_batchnorm']
            use_dropout = True if dropout != 0 else False
            augment_type = config_dict['augment_type']

            print(f"Training with config: {config_dict}")

            # Generate a directory name based on parameter values
            param_str = "_".join([f"{key}-{value}" for key, value in config_dict.items()])
            save_path = os.path.join(experiment_dir, param_str)
            os.makedirs(save_path, exist_ok=True)

            # Initialize datasets
            train_dataset, val_dataset, test_dataset = initialize_datasets(
                train_data, val_data, test_data, augment_type=augment_type
            )

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize and train the model
            model = initialize_model(
                network_size, use_batchnorm, use_dropout, dropout, device
            )

            start_time = time.time()
            epochs_run, train_losses, val_losses = train_model(
                model, train_loader, val_loader, test_loader, device=device,
                num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, save_path=save_path
            )
            elapsed_time = time.time() - start_time

            # Evaluate and log results
            test_accuracy = evaluate_classification_accuracy(model, test_loader, device)
            results[tuple(sorted(config_dict.items()))] = log_results(
                test_accuracy, elapsed_time, train_losses, val_losses, model, epochs_run, save_path, config_dict
            )

            # Update tqdm progress
            pbar.update(1)

    return results


def grid_search_triplet(dataset_path, param_grid , experiment_dir = './grid_search_triplet'):
    # dataset_path = download_dataset(git_url, dataset_filename)
    train_triplets, val_triplets, val_data, test_data = load_triplets_data(dataset_path)
    os.makedirs(experiment_dir, exist_ok=True)

    # Define grid search parameters
    param_combinations = list(itertools.product(*param_grid.values()))
    results = {}

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        config_name = "_".join(f"{key}={value}" for key, value in sorted(param_dict.items()))
        config_save_path = os.path.join(experiment_dir, config_name)
        os.makedirs(config_save_path, exist_ok=True)

        print(f"Testing configuration: {param_dict}")

        # Initialize datasets and dataloaders
        resize_value = param_dict['resize_value']
        train_triplet_dataset, val_triplet_dataset, val_pair_dataset, test_dataset = initialize_datasets_triplet(
            train_triplets, val_triplets, val_data, test_data, resize_value=resize_value
        )

        train_triplet_loader = DataLoader(train_triplet_dataset, batch_size=param_dict['batch_size'], shuffle=True)
        val_triplet_loader = DataLoader(val_triplet_dataset, batch_size=param_dict['batch_size'], shuffle=False)
        val_pair_loader = DataLoader(val_pair_dataset, batch_size=param_dict['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=param_dict['batch_size'], shuffle=False)

        # Initialize model
        model = SiameseNetworkTriplet(
            input_size=(1, resize_value, resize_value),
            embedding_size=param_dict['embedding_size'], use_dropout=param_dict['dropout'], use_batchnorm=param_dict['batchnorm'], network_size=param_dict['network_size']
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Train model and measure time
        start_time = time.time()
        epochs_run, train_losses, val_losses = train_triplet_model(
            model=model,
            train_loader=train_triplet_loader,
            val_triplet_loader=val_triplet_loader,
            val_pair_loader=val_pair_loader,
            test_loader=test_loader,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_epochs=param_dict['num_epochs'],
            lr=param_dict['learning_rate'],
            weight_decay=param_dict['weight_decay'],
            patience=param_dict['patience'],
            min_delta=0.1,
            save_path=config_save_path,
            margin=1.0
        )
        elapsed_time = time.time() - start_time

        # Evaluate on the test set
        test_accuracy = evaluate_classification_accuracy(model, test_loader, 'cuda' if torch.cuda.is_available() else 'cpu')

        # Log results
        results[tuple(sorted(param_dict.items()))] = log_results(
            test_accuracy, elapsed_time, train_losses, val_losses, model, epochs_run, config_save_path, param_dict
        )

    # Save overall results
    overall_results_path = os.path.join(experiment_dir, "overall_results.json")
    with open(overall_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Overall results saved to {overall_results_path}")