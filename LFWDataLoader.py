import os
import time
import json
import pickle
from collections import Counter
import itertools

# Third-party library imports
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from collections import defaultdict


class ImageDataLoader:
    """
    Class for loading, preprocessing, and visualizing image data.
    """

    def __init__(self, data_directory, output_file, random_seed=42):
        """
        Initializes the data loader.

        :param data_directory: Path to the directory containing the image dataset.
        :param output_file: Path to save the processed data as a pickle file.
        :param random_seed: Seed for random operations to ensure reproducibility.
        """
        self.data_directory = data_directory
        self.output_file = output_file
        self.random_seed = random_seed

    def _load_and_preprocess_image(self, image_path, resize_dim=(105, 105), grayscale=True):
        """
        Load and preprocess an image.

        :param image_path: Path to the image file.
        :param resize_dim: Tuple specifying the dimensions to resize the image to.
        :param grayscale: Whether to convert the image to grayscale.
        :return: Numpy array of the preprocessed image.
        """
        try:
            image = Image.open(image_path)
            image = image.resize(resize_dim)
            if grayscale:
                image = image.convert('L')
            image_data = np.asarray(image, dtype='float32') / 255.0  # Normalize pixel values
            if grayscale:
                image_data = image_data[..., np.newaxis]  # Add channel dimension for grayscale
            return image_data
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _generate_image_path(self, person, image_number):
        """
        Generate the file path for an image given the person and image number.

        :param person: Name of the person in the dataset.
        :param image_number: Image number.
        :return: Full path to the image file.
        """
        formatted_number = f"{int(image_number):04d}"
        return os.path.join(self.data_directory, person, f"{person}_{formatted_number}.jpg")

    def _load_image_pair(self, person1, image1, person2, image2, is_identical):
        """
        Load a pair of images and their label.

        :param person1: Name of the first person.
        :param image1: Image number for the first person.
        :param person2: Name of the second person.
        :param image2: Image number for the second person.
        :param is_identical: Whether the images are of the same person (1) or different (0).
        :return: Tuple of two images and their label.
        """
        image_path1 = self._generate_image_path(person1, image1)
        image_path2 = self._generate_image_path(person2, image2)

        img1 = self._load_and_preprocess_image(image_path1)
        img2 = self._load_and_preprocess_image(image_path2)

        if img1 is not None and img2 is not None:
            return img1, img2, is_identical
        else:
            return None, None, None

    def load_data_from_file(self, pairs_file):
        """
        Load data pairs from the provided text file.

        :param pairs_file: Path to the file containing pairs of images.
        :return: Lists of image pairs and their labels.
        """
        x_first, x_second, labels = [], [], []

        with open(pairs_file, 'r') as file:
            lines = file.readlines()[1:]  # Skip the header line if present

        for line in tqdm(lines, desc="Processing pairs"):
            line_parts = line.strip().split()

            if len(line_parts) == 3:  # Identical pairs
                person, img1, img2 = line_parts
                img1_data, img2_data, label = self._load_image_pair(person, img1, person, img2, is_identical=1)
            elif len(line_parts) == 4:  # Non-identical pairs
                person1, img1, person2, img2 = line_parts
                img1_data, img2_data, label = self._load_image_pair(person1, img1, person2, img2, is_identical=0)
            else:
                continue

            if img1_data is not None and img2_data is not None:
                x_first.append(img1_data)
                x_second.append(img2_data)
                labels.append(label)

        return x_first, x_second, labels

    def load_dataset(self, train_pairs_file, test_pairs_file, validation_split=0.2):
        """
        Load and preprocess the training, validation, and test datasets.

        :param train_pairs_file: Path to the training pairs file.
        :param test_pairs_file: Path to the test pairs file.
        :param validation_split: Proportion of the training data to use for validation.
        :return: None, saves the processed data to the output file.
        """
        print("Loading training data...")
        x_first_train, x_second_train, y_train = self.load_data_from_file(train_pairs_file)

        print("Splitting training data into training and validation sets...")
        x_first_train, x_first_val, x_second_train, x_second_val, y_train, y_val = train_test_split(
            x_first_train, x_second_train, y_train, test_size=validation_split, random_state=self.random_seed
        )

        print("Loading test data...")
        x_first_test, x_second_test, y_test = self.load_data_from_file(test_pairs_file)

        data = {
            'train': {'x_first': x_first_train, 'x_second': x_second_train, 'labels': y_train},
            'validation': {'x_first': x_first_val, 'x_second': x_second_val, 'labels': y_val},
            'test': {'x_first': x_first_test, 'x_second': x_second_test, 'labels': y_test}
        }

        print("Saving processed data to file...")
        with open(self.output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved to {self.output_file}")
        return self.output_file

    def generate_statistics(self, stats_file="dataset_statistics.txt"):
        """
        Generate statistics for the dataset and save them to a file.

        :param stats_file: Path to save the statistics.
        :return: None, prints and saves the statistics.
        """
        with open(self.output_file, 'rb') as f:
            data = pickle.load(f)

        stats = {}
        for split in ['train', 'validation', 'test']:
            x_first = data[split]['x_first']
            labels = data[split]['labels']

            total_samples = len(labels)
            unique_labels, label_counts = np.unique(labels, return_counts=True)

            stats[split] = {
                'total_samples': total_samples,
                'class_distribution': dict(zip(unique_labels, label_counts))
            }

            print(f"Statistics for {split.capitalize()} Set:")
            print(f"Total Samples: {total_samples}")
            for label, count in zip(unique_labels, label_counts):
                print(f"Class {label}: {count} samples")

        with open(stats_file, 'w') as f:
            for split, split_stats in stats.items():
                f.write(f"Statistics for {split.capitalize()} Set:\n")
                f.write(f"Total Samples: {split_stats['total_samples']}\n")
                for label, count in split_stats['class_distribution'].items():
                    f.write(f"Class {label}: {count} samples\n")
                f.write("\n")

        print(f"Statistics saved to {stats_file}")


    def visualize_sample_pairs(self, split='train', sample_count=5):
        """
        Visualize a few image pairs from the dataset.

        :param split: Dataset split to visualize ('train', 'validation', or 'test').
        :param sample_count: Number of sample pairs to visualize.
        :return: None
        """
        with open(self.output_file, 'rb') as f:
            data = pickle.load(f)

        x_first = data[split]['x_first']
        x_second = data[split]['x_second']
        labels = data[split]['labels']

        for i in range(min(sample_count, len(labels))):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(x_first[i].squeeze(), cmap='gray')
            axes[0].set_title("First Image")
            axes[1].imshow(x_second[i].squeeze(), cmap='gray')
            axes[1].set_title("Second Image")
            plt.suptitle(f"Label: {'Identical' if labels[i] == 1 else 'Non-Identical'}")
            plt.show()

class TripletDataLoader:
    def __init__(self, data_directory, output_file, random_seed=42):
        self.data_directory = data_directory
        self.output_file = output_file
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def _load_and_preprocess_image(self, image_path, resize_dim=(105, 105), grayscale=True):
        """
        Load and preprocess an image.

        :param image_path: Path to the image file.
        :param resize_dim: Tuple specifying the dimensions to resize the image to.
        :param grayscale: Whether to convert the image to grayscale.
        :return: Numpy array of the preprocessed image.
        """
        try:
            image = Image.open(image_path)
            image = image.resize(resize_dim)
            if grayscale:
                image = image.convert('L')
            image_data = np.asarray(image, dtype='float32') / 255.0  # Normalize pixel values
            if grayscale:
                image_data = image_data[..., np.newaxis]  # Add channel dimension for grayscale
            return image_data
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _generate_image_path(self, person, image_number):
        """
        Generate the file path for an image given the person and image number.

        :param person: Name of the person in the dataset.
        :param image_number: Image number.
        :return: Full path to the image file.
        """
        formatted_number = f"{int(image_number):04d}"
        return os.path.join(self.data_directory, person, f"{person}_{formatted_number}.jpg")

    def _load_image_pair(self, person1, image1, person2, image2, is_identical):
        """
        Load a pair of images and their label.

        :param person1: Name of the first person.
        :param image1: Image number for the first person.
        :param person2: Name of the second person.
        :param image2: Image number for the second person.
        :param is_identical: Whether the images are of the same person (1) or different (0).
        :return: Tuple of two images and their label.
        """
        image_path1 = self._generate_image_path(person1, image1)
        image_path2 = self._generate_image_path(person2, image2)

        img1 = self._load_and_preprocess_image(image_path1)
        img2 = self._load_and_preprocess_image(image_path2)

        if img1 is not None and img2 is not None:
            return img1, img2, is_identical
        else:
            return None, None, None

    def load_data_from_file(self, pairs_file):
        """
        Load data pairs from the provided text file.

        :param pairs_file: Path to the file containing pairs of images.
        :return: Lists of image pairs and their labels.
        """
        x_first, x_second, labels = [], [], []

        with open(pairs_file, 'r') as file:
            lines = file.readlines()[1:]  # Skip the header line if present

        for line in tqdm(lines, desc="Processing pairs"):
            line_parts = line.strip().split()

            if len(line_parts) == 3:  # Identical pairs
                person, img1, img2 = line_parts
                img1_data, img2_data, label = self._load_image_pair(person, img1, person, img2, is_identical=1)
            elif len(line_parts) == 4:  # Non-identical pairs
                person1, img1, person2, img2 = line_parts
                img1_data, img2_data, label = self._load_image_pair(person1, img1, person2, img2, is_identical=0)
            else:
                continue

            if img1_data is not None and img2_data is not None:
                x_first.append(img1_data)
                x_second.append(img2_data)
                labels.append(label)

        return x_first, x_second, labels

    def _group_images_by_class(self):
        grouped_images = defaultdict(list)
        for person_dir in os.listdir(self.data_directory):
            person_path = os.path.join(self.data_directory, person_dir)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_path, img_file)
                        grouped_images[person_dir].append(img_path)
        return grouped_images

    def _create_triplets(self, grouped_images):
        triplets = []
        classes = list(grouped_images.keys())
        for anchor_class, images in grouped_images.items():
            if len(images) < 2:
                continue  # Skip classes with insufficient samples

            negative_classes = [cls for cls in classes if cls != anchor_class]
            for anchor_img_path in images:
                # Select a positive sample (ensure it is not the same as the anchor)
                positive_img_path = anchor_img_path
                while positive_img_path == anchor_img_path:
                    positive_img_path = np.random.choice(images)

                # Select a negative sample from a different class
                negative_class = np.random.choice(negative_classes)
                negative_img_path = np.random.choice(grouped_images[negative_class])

                # Load and preprocess images
                anchor_img = self._load_and_preprocess_image(anchor_img_path)
                positive_img = self._load_and_preprocess_image(positive_img_path)
                negative_img = self._load_and_preprocess_image(negative_img_path)

                # Ensure all images are valid
                if anchor_img is not None and positive_img is not None and negative_img is not None:
                    triplets.append((anchor_img, positive_img, negative_img))

        print(f"Total triplets created: {len(triplets)}")  # Debug: Log the number of triplets
        # Log a few sample triplets for debugging
        if triplets:
            print("Sample triplets (array shapes):")
            for i, triplet in enumerate(triplets[:3]):
                anchor, positive, negative = triplet
                print(f"Triplet {i + 1}: Anchor {anchor.shape}, Positive {positive.shape}, Negative {negative.shape}")
        return triplets

    def load_dataset(self, train_pairs_file, test_pairs_file, validation_split=0.2):
        print("Grouping images by class...")
        grouped_images = self._group_images_by_class()

        # Split the grouped images into train and validation sets
        val_size = int(len(grouped_images) * validation_split)
        grouped_val_images = {k: grouped_images[k] for k in list(grouped_images.keys())[:val_size]}
        grouped_train_images = {k: grouped_images[k] for k in list(grouped_images.keys())[val_size:]}

        print("Creating training triplets...")
        train_triplets = self._create_triplets(grouped_train_images)

        print("Creating validation triplets...")
        val_triplets = self._create_triplets(grouped_val_images)

        print("Loading training data...")
        x_first_train, x_second_train, y_train = self.load_data_from_file(train_pairs_file)

        print("Splitting training data into training and validation sets...")
        x_first_train, x_first_val, x_second_train, x_second_val, y_train, y_val = train_test_split(
            x_first_train, x_second_train, y_train, test_size=validation_split, random_state=self.random_seed
        )

        print("Loading test data...")
        x_first_test, x_second_test, y_test = self.load_data_from_file(test_pairs_file)

        data = {
            'triplets': {'val': train_triplets , 'train': val_triplets},
            'validation': {'x_first': x_first_val, 'x_second': x_second_val, 'labels': y_val},
            'test': {'x_first': x_first_test, 'x_second': x_second_test, 'labels': y_test},
        }

        print("Saving processed data to file...")
        with open(self.output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved to {self.output_file}")
        return self.output_file


    def visualize_triplet(self, triplet):
        anchor, positive, negative = triplet
        anchor_img = np.squeeze(anchor)  # Remove channel dimension for visualization
        positive_img = np.squeeze(positive)
        negative_img = np.squeeze(negative)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(anchor_img, cmap='gray')
        plt.title("Anchor")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(positive_img, cmap='gray')
        plt.title("Positive")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(negative_img, cmap='gray')
        plt.title("Negative")
        plt.axis("off")

        plt.show()
