# **Siamese and Triplet Network for Image Similarity Learning**

This repository implements **Siamese Networks** as presented in the paper [*"Siamese Neural Networks for One-shot Image Recognition"*](https://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf) by Koch et al., with an **extension to Triplet Loss**. The addition of Triplet Loss allows the model to learn embeddings that preserve relative similarity relationships, useful for tasks like image retrieval and clustering.

---

## **Features**

- **Siamese Network**: Implements the classic Siamese architecture using binary cross-entropy loss for pairwise similarity prediction.
- **Triplet Network Extension**: Allows the use of Triplet Margin Loss for embedding learning using anchor-positive-negative triplets.
- **End-to-End Process**: Full pipeline for data loading, preprocessing, training, and evaluation.
- **Grid Search for Hyperparameter Tuning**: Supports hyperparameter tuning for both the **Siamese Network** and the **Triplet Network** extensions.

---

## **Repository Structure**

- `dataset.py` - Utilities for creating datasets for **Siamese** and **Triplet** training.
- `LFWDataLoader.py` - Loading and preprocessing the Labeled Faces in the Wild (LFW) dataset.
- `model.py` - Model architectures for **Siamese Network** and **Triplet Network**.
- `train.py` - Training pipelines for both networks.
- `utils.py` - Helper functions for logging, saving results, and evaluation.
- `experiments.ipynb` - **Jupyter Notebook** performing the complete workflow: 
    - Data Loading
    - Hyperparameter Grid Search
    - Training and Evaluation of both **Siamese** and **Triplet Network** models.

---

## **Experiments.ipynb**

The **`experiments.ipynb`** notebook demonstrates the end-to-end process:

1. **Data Preparation**:
   - Load and preprocess the dataset using utilities provided in `LFWDataLoader.py`.
   - Create datasets for **Siamese Pairs** and **Triplets**.

2. **Grid Search**:
   - Perform hyperparameter tuning for both the **Siamese Network** (using binary cross-entropy loss) and the **Triplet Network** (using Triplet Margin Loss).

3. **Training and Evaluation**:
   - Train the models using optimized hyperparameters.
   - Evaluate the performance on the test dataset for both approaches.

4. **Results Visualization**:
   - Visualize training loss, accuracy, and predictions.

---
## **Setup**

### **Requirements**
Install dependencies:
```bash
pip install -r requirements.txt
```

### **Dataset**
The project uses the Labeled Faces in the Wild (LFW) dataset. To use this project:
1. Download the dataset from the [LFW website](http://vis-www.cs.umass.edu/lfw/).
2. Organize the dataset directory as follows:
   ```
   dataset/
   ├── person1/
   │   ├── person1_0001.jpg
   │   ├── person1_0002.jpg
   ├── person2/
   │   ├── person2_0001.jpg
   │   ├── person2_0002.jpg
   ```

### **Dataset Preprocessing**
Run the `ImageDataLoader` to preprocess the dataset:
```python
from LFWDataLoader import ImageDataLoader

loader = ImageDataLoader(data_directory="path/to/dataset", output_file="processed_data.pkl")
loader.load_dataset(train_pairs_file="train_pairs.txt", test_pairs_file="test_pairs.txt")
```

---

## **How to Train the Siamese Model**

1. **Prepare the Dataset**
   Use the `initialize_datasets` function from `dataset.py` to create datasets:
   ```python
   from dataset import initialize_datasets
   train_data, val_data, test_data = load_data("processed_data.pkl")
   train_dataset, val_dataset, test_dataset = initialize_datasets(train_data, val_data, test_data)
   ```

2. **Create DataLoaders**
   ```python
   from torch.utils.data import DataLoader

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
   test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
   ```

3. **Train the Model**
   Use the `train_model` function in `train.py`:
   ```python
   from train import train_model
   from model import SiameseNetwork

   model = SiameseNetwork().to("cuda")
   train_model(model, train_loader, val_loader, test_loader, device="cuda", num_epochs=10, lr=0.001)
   ```

4. **Save Results**
   Training logs, model weights, and loss plots are saved in the `model_logs` directory.

---

## **How to Train the Triplet Model**

1. **Prepare the Dataset**
   Use the `initialize_datasets_triplet` function:
   ```python
   from dataset import initialize_datasets_triplet
   train_triplets, val_triplets, val_data, test_data = load_triplets_data("processed_data.pkl")
   train_triplet_dataset, val_triplet_dataset, val_pair_dataset, test_dataset = initialize_datasets_triplet(
       train_triplets, val_triplets, val_data, test_data
   )
   ```

2. **Create DataLoaders**
   ```python
   from torch.utils.data import DataLoader

   train_loader = DataLoader(train_triplet_dataset, batch_size=32, shuffle=True)
   val_triplet_loader = DataLoader(val_triplet_dataset, batch_size=32, shuffle=False)
   val_pair_loader = DataLoader(val_pair_dataset, batch_size=32, shuffle=False)
   test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
   ```

3. **Train the Model**
   ```python
   from train import train_triplet_model
   from model import SiameseNetworkTriplet

   model = SiameseNetworkTriplet().to("cuda")
   train_triplet_model(model, train_loader, val_triplet_loader, val_pair_loader, test_loader, device="cuda", num_epochs=10)
   ```

---

## **Run Experiments**

### **Grid Search**
Perform hyperparameter tuning with `grid_search_siemse` or `grid_search_triplet`:
```python
from train import grid_search_siemse

param_grid = {
    "batch_size": [16, 32],
    "lr": [0.001, 0.0001],
    "weight_decay": [0.0005, 0.001],
    "network_size": ["small", "medium"],
    "dropout": [0, 0.5],
    "augment_type": ["none", "basic"]
}

grid_search_siemse("processed_data.pkl", device="cuda", param_grid=param_grid, num_epochs=10)
```

### **Visualize Results**
Loss plots and logs are saved in the `grid_search` directory.

---

## **Acknowledgements**
- PyTorch for deep learning framework.
- LFW dataset for image pairs.
- Authors of the original Siamese and Triplet network papers.

## **License**
This project is licensed under the MIT License.
