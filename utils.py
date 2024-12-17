import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def log_epoch(epoch, num_epochs, train_loss, val_loss, metrics):
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {metrics['val_accuracy']:.4f}, "
        f"ROC AUC: {metrics['roc_auc']:.4f}, Best Threshold Accuracy: {metrics['best_threshold_accuracy']:.4f}, "
        f"Triplet Accuracy: {metrics['triplet_accuracy']:.4f}"
    )

def log_test_results(metrics):
    print(
        f"Test Accuracy: {metrics['test_accuracy']:.4f}, "
        f"Test ROC AUC: {metrics['test_roc_auc']:.4f}, "
        f"Test Best Threshold Accuracy: {metrics['test_best_threshold_accuracy']:.4f}"
    )

# Save train val learning curves Plots
def plot_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

def save_logs(logs, save_path):
    with open(os.path.join(save_path, "logs.json"), 'w') as f:
        json.dump(logs, f)
    print(f"Logs saved to {save_path}/logs.json")


# Classification Accuracy Function (return accuracy)
def evaluate_classification_accuracy(model, loader, device ='cpu', triplet=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            if triplet:
                outputs = model.forward_pair(x1, x2)
            else:
                outputs = model(x1, x2)
            predictions = (outputs.squeeze() > 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)
    return correct / total

def compute_roc_auc_and_threshold(model, dataloader, device='cpu'):
    model.eval()
    distances, labels = [], []
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            embedding1 = model(x1)
            embedding2 = model(x2)
            distance = torch.norm(embedding1 - embedding2, dim=1).cpu().numpy()
            distances.extend(distance)
            labels.extend(y.cpu().numpy())
    roc_auc = roc_auc_score(labels, distances)
    # Find the best threshold
    thresholds = torch.linspace(0, max(distances), steps=100)
    best_threshold, best_accuracy = 0, 0
    for threshold in thresholds:
        predictions = (torch.tensor(distances) < threshold).float()
        accuracy = (predictions == torch.tensor(labels)).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return roc_auc, best_threshold, best_accuracy

def evaluate_model(model, loader, criterion, device='cpu', triplet=False):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            if triplet:
                outputs = model.forward_pair(x1, x2)
            else:
                outputs = model(x1, x2)
            loss = criterion(outputs.squeeze(), y)
            val_loss += loss.item()
    return val_loss / len(loader)

def validate_model_triplet(model, val_pair_loader, val_triplet_loader, device):
    val_loss = evaluate_model(model, val_pair_loader, nn.BCELoss(), device, triplet=True)

    val_accuracy = evaluate_classification_accuracy(model, val_pair_loader, device, triplet=True)
    roc_auc, best_threshold, best_threshold_accuracy = compute_roc_auc_and_threshold(model, val_pair_loader, device)
    triplet_accuracy = evaluate_triplet_accuracy(model, val_triplet_loader, device)

    val_metrics = {
        "val_accuracy": val_accuracy,
        "roc_auc": roc_auc,
        "best_threshold": best_threshold.item(),
        "best_threshold_accuracy": best_threshold_accuracy,
        "triplet_accuracy": triplet_accuracy
    }

    return val_loss, val_metrics

def validate_model(model, val_loader, criterion, device):
    val_loss = evaluate_model(model, val_loader, criterion, device)
    val_accuracy = evaluate_classification_accuracy(model, val_loader, device)
    return val_loss, val_accuracy

def log_epoch_triplet(epoch, num_epochs, train_loss, val_loss, metrics):
    print(metrics.keys())
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {metrics['val_accuracy']:.4f}, "
        f"ROC AUC: {metrics['roc_auc']:.4f}, Best Threshold Accuracy: {metrics['best_threshold_accuracy']:.4f}, "
        f"Triplet Accuracy: {metrics['triplet_accuracy']:.4f}"
    )
def log_epoch(epoch, num_epochs, train_loss, val_loss, metrics):
    print(metrics.keys())
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {metrics['val_accuracy']:.4f}, "
    )
    
def early_stopping(val_loss, best_val_loss, min_delta, patience_counter, patience):
    if val_loss < best_val_loss - min_delta:
        return True
    return False

def evaluate_test_model(model, test_loader, device, triplet=False):
    roc_auc, best_threshold, best_threshold_accuracy = compute_roc_auc_and_threshold(model, test_loader, device)
    test_accuracy = evaluate_classification_accuracy(model, test_loader, device, triplet=triplet)
    return {
        "test_accuracy": test_accuracy,
        "test_roc_auc": roc_auc,
        "test_best_threshold_accuracy": best_threshold_accuracy
    }

def log_test_results_triplet(metrics):
    print(
        f"Test Accuracy: {metrics['test_accuracy']:.4f}, "
        f"Test ROC AUC: {metrics['test_roc_auc']:.4f}, "
        f"Test Best Threshold Accuracy: {metrics['test_best_threshold_accuracy']:.4f}"
    )
    
def log_test_results(metrics):
    print(
        f"Test Accuracy: {metrics['test_accuracy']:.4f}, "
    )

def evaluate_triplet_accuracy(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            pos_distance = torch.norm(anchor_embedding - positive_embedding, dim=1)
            neg_distance = torch.norm(anchor_embedding - negative_embedding, dim=1)
            correct += (pos_distance < neg_distance).sum().item()
            total += len(anchor)
    return correct / total

# Save Validation Images with TP, TN, FP, FN
def save_validation_images(model, val_loader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    tp_saved, tn_saved, fp_saved, fn_saved = False, False, False, False

    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2)
            predictions = (outputs.squeeze() > 0.5).float()

            for j in range(x1.size(0)):
                label = y[j].item()
                prediction = predictions[j].item()

                if label == 1 and prediction == 1 and not tp_saved:  # True Positive
                    save_image_pair(x1[j], x2[j], "TP", save_dir)
                    tp_saved = True
                elif label == 0 and prediction == 0 and not tn_saved:  # True Negative
                    save_image_pair(x1[j], x2[j], "TN", save_dir)
                    tn_saved = True
                elif label == 0 and prediction == 1 and not fp_saved:  # False Positive
                    save_image_pair(x1[j], x2[j], "FP", save_dir)
                    fp_saved = True
                elif label == 1 and prediction == 0 and not fn_saved:  # False Negative
                    save_image_pair(x1[j], x2[j], "FN", save_dir)
                    fn_saved = True

                if tp_saved and tn_saved and fp_saved and fn_saved:
                    return

def save_image_pair(image1, image2, label, save_dir):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image1.squeeze().cpu(), cmap='gray')
    plt.title("Image 1")
    plt.subplot(1, 2, 2)
    plt.imshow(image2.squeeze().cpu(), cmap='gray')
    plt.title("Image 2")
    plt.suptitle(f"Category: {label}")
    plt.savefig(os.path.join(save_dir, f"{label}.png"))
    plt.close()
    
def log_results(test_accuracy, elapsed_time, train_losses, val_losses, model, epochs_run, save_path, config_dict):
    """Log performance metrics for a specific configuration."""
    results = {
        "test_accuracy": test_accuracy,
        "training_time": elapsed_time,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs_run": epochs_run,
        "config": config_dict
    }
    results_path = os.path.join(save_path, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    return results