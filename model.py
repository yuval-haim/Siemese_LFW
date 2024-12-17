import torch
import torch.nn as nn

# Siamese Network Architecture
class SiameseNetwork(nn.Module):
    # Architecture for the Siamese Network - same as described in the paper
    def __init__(self, input_shape=(1, 105, 105), use_batchnorm=True, use_dropout=True, network_size="small", dropout_rate=0.5):
        super(SiameseNetwork, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        filters = {
            "small": [16, 32, 64, 128],
            "medium": [32, 64, 128, 256],
            "large": [64, 128, 256, 512]
        }[network_size]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=10, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[0]) if use_batchnorm else nn.Identity(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters[0], filters[1], kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[1]) if use_batchnorm else nn.Identity(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters[1], filters[2], kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[2]) if use_batchnorm else nn.Identity(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters[2], filters[3], kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[3]) if use_batchnorm else nn.Identity()
        )

        fc_input_size = filters[3] * 6 * 6
        fc_output_size = {
            "small": 512,
            "medium": 2048,
            "large": 4096
        }[network_size]

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, fc_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        )

        self.output_layer = nn.Linear(fc_output_size, 1)

    def forward_once(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        distance = torch.abs(output1 - output2)
        output = torch.sigmoid(self.output_layer(distance))
        return output


class SiameseNetworkTriplet(nn.Module):
    # Architecture for the Siamese Network extended to support triplet loss
    def __init__(self, input_size=(1, 105, 105), embedding_size=128, use_dropout=True, dropout_prob=0.5, use_batchnorm=True, network_size="large"):
        """
        Initializes the SiameseNetworkTriplet with configurable network size, BatchNorm, and Dropout.

        Args:
            input_size (tuple): Input image size as (channels, height, width). Default is (1, 105, 105).
            embedding_size (int): Size of the output embedding. Default is 128.
            use_dropout (bool): Whether to use Dropout layers in fully connected layers. Default is True.
            dropout_prob (float): Probability of dropout if Dropout is used. Default is 0.5.
            use_batchnorm (bool): Whether to use Batch Normalization layers. Default is True.
            network_size (str): Network size: 'small', 'medium', or 'large'. Default is 'medium'.
        """
        super(SiameseNetworkTriplet, self).__init__()
        
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.network_size = network_size

        # Define filter sizes based on network size
        filter_sizes = self._get_filter_sizes(network_size)

        # Convolutional feature extractor
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_size[0], filter_sizes[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(filter_sizes[0]) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size: (filter_sizes[0], H/2, W/2)
            
            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(filter_sizes[1]) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size: (filter_sizes[1], H/4, W/4)
            
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_sizes[2]) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output size: (filter_sizes[2], H/8, W/8)
        )
        
        # Fully connected layers with Dropout option
        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_flattened_size(input_size, filter_sizes), 512),
            nn.BatchNorm1d(512) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob) if use_dropout else nn.Identity(),  # Dropout applied here only
            nn.Linear(512, embedding_size)
        )

        # Output layer for pair-based evaluations
        self.output_layer = nn.Linear(embedding_size, 1)

    def _get_filter_sizes(self, network_size):
        """
        Returns the filter sizes for each convolutional layer based on the network size.

        Args:
            network_size (str): Network size ('small', 'medium', 'large').

        Returns:
            list: Filter sizes for each convolutional layer.
        """
        if network_size == "small":
            return [16, 32, 64]
        elif network_size == "medium":
            return [32, 64, 128]
        elif network_size == "large":
            return [64, 128, 256]
        else:
            raise ValueError("Invalid network size. Choose from 'small', 'medium', 'large'.")

    def _get_flattened_size(self, input_size, filter_sizes):
        """
        Dynamically calculates the flattened size after the convolutional layers.
        
        Args:
            input_size (tuple): Input size (channels, height, width).
            filter_sizes (list): List of filter sizes used in the convolutional layers.
        
        Returns:
            int: Flattened size after convolutional layers.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)  # Create a dummy input with batch size = 1
            output = self.conv_net(dummy_input)       # Pass through conv_net
            flattened_size = output.numel()          # Get the total number of elements in the output
        return flattened_size

    def forward(self, x):
        """
        Forward pass for single input (triplet-based training).
        """
        x = self.conv_net(x)
        x = self.fc_net(x)
        return x

    def forward_pair(self, x1, x2):
        """
        Forward pass for pair-based evaluation.
        Computes similarity score (e.g., Contrastive Loss style).
        """
        embedding1 = self.forward(x1)
        embedding2 = self.forward(x2)

        # Compute absolute difference
        distance = torch.abs(embedding1 - embedding2)

        # Predict similarity score
        output = torch.sigmoid(self.output_layer(distance))
        return output
