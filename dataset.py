import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image
import pickle

class SiameseDataset(Dataset):
    def __init__(self, x_first, x_second, labels, transform=None):
        self.x_first = torch.tensor(np.transpose(np.array(x_first), (0, 3, 1, 2)), dtype=torch.float32)
        self.x_second = torch.tensor(np.transpose(np.array(x_second), (0, 3, 1, 2)), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __getitem__(self, idx):
        x1 = self.x_first[idx]
        x2 = self.x_second[idx]
        label = self.labels[idx]

        # Convert numpy arrays to PIL Images
        x1 = tensor_to_pil(x1)
        x2 = tensor_to_pil(x2)
        # Apply transformations
        if self.transform:
            x1 = self.transform(x1)  # ToTensor will convert to [C, H, W]
            x2 = self.transform(x2)

        return (x1, x2, label)

    def __len__(self):
        return len(self.labels)


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, idx):
        # Extract anchor, positive, and negative
        anchor, positive, negative = self.triplets[idx]

        # Convert numpy arrays to PIL Images
        anchor = Image.fromarray((anchor.squeeze() * 255).astype('uint8'))
        positive = Image.fromarray((positive.squeeze() * 255).astype('uint8'))
        negative = Image.fromarray((negative.squeeze() * 255).astype('uint8'))

        # Apply transformations
        if self.transform:
            anchor = self.transform(anchor)  # ToTensor will convert to [C, H, W]
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.triplets)

def get_transform(resize_value=128):
    return transforms.Compose([
        transforms.Resize((resize_value, resize_value)),  # Resize to the specified value
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])


def initialize_datasets_triplet(train_triplets, val_triplets, val_data, test_data, resize_value=128):
    """Initialize train, validation, and test datasets with resizing and augmentation options."""
    
    transform = get_transform(resize_value=resize_value)

    train_triplet_dataset = TripletDataset(train_triplets, transform=transform)
    val_triplet_dataset = TripletDataset(val_triplets, transform=transform)

    val_pair_dataset = SiameseDataset(
        x_first=val_data['x_first'],
        x_second=val_data['x_second'],
        labels=val_data['labels'],
        transform=transform
    )
    test_dataset = SiameseDataset(
        x_first=test_data['x_first'],
        x_second=test_data['x_second'],
        labels=test_data['labels'],
        transform=transform
    )

    return train_triplet_dataset, val_triplet_dataset, val_pair_dataset, test_dataset

# Augmentation Pipeline


# Helper Wrapper for Augmented Samples
class DatasetWrapper(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Function to Extend the Dataset (Oversampling with Augmentations)
def extend_dataset_with_augmentations(dataset, transform):
    augmented_samples = []
    for i in range(len(dataset)):
        original = dataset[i]
        if isinstance(original, tuple) and len(original) == 2:
            (x1, x2, label), (x1_aug, x2_aug, label_aug) = original
            augmented_samples.append((x1, x2, label))
            augmented_samples.append((x1_aug, x2_aug, label_aug))
        else:
            x1, x2, label = original
            augmented_samples.append((x1, x2, label))
            augmented_samples.append((transform(x1), transform(x2), label))

    # Return the augmented dataset
    return ConcatDataset([DatasetWrapper(augmented_samples)])

def get_transform_siemse(augment_type="none", resize_value=105):
    """
    Returns a transformation pipeline for Siamese dataset images.

    Args:
        augment_type (str): Type of augmentation to apply ("none", "basic", or "oversample").
        resize_value (int): Size to resize images for input to the model.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    if augment_type in ["basic", "oversample"]:
        return transforms.Compose([
            transforms.Resize((105, 105)),  # Resize to match model input
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to improve model performance
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize_value, resize_value)),  # Resize to the specified value
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

def initialize_datasets(train_data, val_data, test_data, augment_type="none"):
    """
    Initialize train, validation, and test datasets with augmentation and oversampling options for the Siamese training.

    Args:
        train_data (dict): Training data containing 'x_first', 'x_second', and 'labels'.
        val_data (dict): Validation data containing 'x_first', 'x_second', and 'labels'.
        test_data (dict): Test data containing 'x_first', 'x_second', and 'labels'.
        augment_type (str): Type of augmentation to apply ("none", "basic", or "oversample").

    Returns:
        tuple: train_dataset, val_dataset, test_dataset
    """
    train_transform = get_transform_siemse(augment_type=augment_type)
    val_transform = get_transform_siemse(augment_type="none")
    test_transform = get_transform_siemse(augment_type="none")

    train_dataset = SiameseDataset(
        x_first=train_data['x_first'],
        x_second=train_data['x_second'],
        labels=train_data['labels'],
        transform=train_transform
    )

    val_dataset = SiameseDataset(
        x_first=val_data['x_first'],
        x_second=val_data['x_second'],
        labels=val_data['labels'],
        transform=val_transform
    )

    test_dataset = SiameseDataset(
        x_first=test_data['x_first'],
        x_second=test_data['x_second'],
        labels=test_data['labels'],
        transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset


def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL Image.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W).

    Returns:
        PIL.Image: Converted PIL image.
    """
    # If the tensor is normalized, denormalize it (example normalization values)
    # tensor = tensor * std + mean
    # Scale to [0, 255]
    tensor = tensor * 255
    # Ensure uint8 data type
    tensor = tensor.byte()
    # Convert to PIL image
    pil_image = to_pil_image(tensor)
    return pil_image

def load_triplets_data(dataset_path):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    train_triplets = data['triplets']['train']
    val_triplets = data['triplets']['val']
    val_data = data['validation']
    test_data = data['test']
    return train_triplets, val_triplets, val_data, test_data

def load_data(dataset_path):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    train_data = data['train']
    val_data = data['validation']
    test_data = data['test']
    return train_data, val_data, test_data