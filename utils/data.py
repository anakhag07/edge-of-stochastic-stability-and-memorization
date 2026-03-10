from typing import Union, Tuple, Optional, Dict
import pickle
import os

import torch as T
import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F
from utils.measure import extract_feature_matrix, identify_knn_outliers_by_neighbor_mix




def get_dataset_presets():

    dataset_presets = {
            'cifar10_2cls': {
                'input_dim': 3*32*32,
                'output_dim': 2
            },
            'cifar10': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'cifar10_ez': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'svhn': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'fmnist': {
                'input_dim': 1*28*28,
                'output_dim': 10
            },
            'imagenet32': {
                'input_dim': 3*32*32,
                'output_dim': 1000
            }

        }
    
    return dataset_presets



def prepare_cifar10_2cls(dataset_folder: Path, num_data: int, classes: list, dataset_seed: int = 888, loss_type='mse'):
    datafolder = dataset_folder / 'cifar10'
    trainset = datasets.CIFAR10(root=datafolder, train=True, download=False)
    testset = datasets.CIFAR10(root=datafolder, train=False, download=False)

    CLASS1, CLASS2 = classes
    label_map = {CLASS1: 0, CLASS2: 1}
    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset

        idx1 = [i for i, target in enumerate(partition_set.targets) if target == CLASS1]
        idx2 = [i for i, target in enumerate(partition_set.targets) if target == CLASS2]

        if partition == 'train':
            idx1 = idx1[:train_size // 2]
            idx2 = idx2[:train_size // 2]

        idx = idx1 + idx2
        idx.sort()

        partition_data = partition_set.data[idx]
        partition_target = np.array([label_map[t] for t in np.array(partition_set.targets)[idx]])

        X = T.tensor(partition_data).float() / 255.0
        X = X - T.tensor([0.4914, 0.4822, 0.4465])
        X = X / T.tensor([0.2023, 0.1994, 0.2010])
        X = rearrange(X, 'b w h c -> b c w h').detach()

        Y = T.tensor(partition_target, dtype=torch.long)
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=2).float()
        else:
            Y = Y.long()

        if partition == 'train':
            X_train, Y_train = X, Y
        else:
            X_test, Y_test = X, Y

    return X_train, Y_train, X_test, Y_test


def prepare_cifar10(dataset_folder: Path, 
                    num_data: int, 
                    dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    datafolder = dataset_folder / 'cifar10'
    trainset = datasets.CIFAR10(root=datafolder, train=True, download=False)
    testset = datasets.CIFAR10(root=datafolder, train=False, download=False)

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        if partition == 'train':
            rng = np.random.default_rng(dataset_seed)
            idx = rng.choice(len(partition_set), train_size, replace=False)

        else:
            idx = list(range(len(partition_set)))

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        X = X - T.tensor([0.4914, 0.4822, 0.4465])
        X = X / T.tensor([0.2023, 0.1994, 0.2010])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()


        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test

def prepare_fmnist(dataset_folder: Path, 
                            num_data: int, 
                            dataset_seed: int = 888,
                            loss_type: str = 'mse'
                            ):
    datafolder = dataset_folder / 'fmnist'
    trainset = datasets.FashionMNIST(root=datafolder, train=True, download=False)
    testset = datasets.FashionMNIST(root=datafolder, train=False, download=False)

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        if partition == 'train':
            rng = np.random.default_rng(dataset_seed)
            idx = rng.choice(len(partition_set), train_size, replace=False)
        else:
            idx = list(range(len(partition_set)))

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]

        X = partition_data.unsqueeze(1)  # Add channel dimension
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        # Standard normalization for Fashion-MNIST
        X = X - 0.2860
        X = X / 0.3530

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test

def prepare_cifar100(num_data: int):
    pass

def prepare_svhn(dataset_folder: Path, 
                    num_data: int, 
                    dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    import scipy.io as sio
    
    datafolder = dataset_folder / 'svhn'

    svhn_train_path = datafolder / 'train_32x32.mat'
    svhn_test_path = datafolder / 'test_32x32.mat'

    # Load training data
    train_data = sio.loadmat(svhn_train_path)
    X_svhn_train = np.transpose(train_data['X'], (3, 0, 1, 2))  # Convert to (N, H, W, C)
    Y_svhn_train = train_data['y'].reshape(-1)
    # SVHN labels are from 1-10, with 10 representing 0. Convert to 0-9
    Y_svhn_train[Y_svhn_train == 10] = 0

    # Load test data
    test_data = sio.loadmat(svhn_test_path)
    X_svhn_test = np.transpose(test_data['X'], (3, 0, 1, 2))
    Y_svhn_test = test_data['y'].reshape(-1)
    Y_svhn_test[Y_svhn_test == 10] = 0

    # Convert to torch tensors
    X_svhn_train = torch.from_numpy(X_svhn_train).float() / 255.0  # Normalize to [0, 1]
    Y_svhn_train = torch.from_numpy(Y_svhn_train).long()
    X_svhn_test = torch.from_numpy(X_svhn_test).float() / 255.0
    Y_svhn_test = torch.from_numpy(Y_svhn_test).long()


    for partition in ['train', 'test']:
        if partition == 'train':
            X = X_svhn_train
            Y = Y_svhn_train
            # If num_data is specified, limit the training data
            if num_data > 0 and num_data <= len(X_svhn_train):
                rng = np.random.default_rng(dataset_seed)
                idx = rng.choice(len(X_svhn_train), num_data, replace=False)
                X = X[idx]
                Y = Y[idx]
        else:
            X = X_svhn_test[:10_000]
            Y = Y_svhn_test[:10_000]
        


        # Normalize the images
        X = X # THE IMAGES ARE ALREADY NORMALIZED
        X = X.float()

        # Normalize using precomputed statistics
        X = X - T.tensor([0.4377, 0.4438, 0.4728])
        X = X / T.tensor([0.1980, 0.2010, 0.1970])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()


        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
    
    return X_train, Y_train, X_test, Y_test



def prepare_cifar10_ez(dataset_folder: Path, num_data: int, dataset_seed: int = 888, loss_type: str = 'mse'):
    # Define the file paths
    cifar_folder = dataset_folder / 'cifar10_ez'
    train_x_path = cifar_folder / 'X_train_pulled.npy'
    train_y_path = cifar_folder / 'Y_train.npy'
    test_x_path = cifar_folder / 'X_test_pulled.npy'
    test_y_path = cifar_folder / 'Y_test.npy'
    
    # Load the data
    X_train = torch.tensor(np.load(train_x_path)).float()
    Y_train = torch.tensor(np.load(train_y_path))
    X_test = torch.tensor(np.load(test_x_path)).float()
    Y_test = torch.tensor(np.load(test_y_path))
    
    # If num_data is specified, limit the training data
    if num_data > 0 and num_data <= len(X_train):
        rng = np.random.default_rng(dataset_seed)
        idx = rng.choice(len(X_train), num_data, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
    
    # Handle Y based on loss_type
    if loss_type == 'mse':
        Y_train = Y_train.float()
        Y_test = Y_test.float()
        #F.one_hot(Y_test.long(), num_classes=10).float()
    else:
        raise NotImplementedError("Cross-entropy loss is not supported for CIFAR-10 EZ dataset YET")
        Y_train = Y_train.long()
        Y_test = Y_test.long()
    
    return X_train, Y_train, X_test, Y_test


def load_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read one ImageNet-32 pickle batch and return images (HWC uint8) & labels."""
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")

    data = entry["data"]          # (N, 3072) uint8
    labels = entry["labels"]

    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(labels, dtype=np.int64)
    return images, labels


def load_imagenet32(root: str, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load all train batches or the validation split into memory."""
    if train:
        batches = sorted(f for f in os.listdir(root) if f.startswith("train_data_batch_"))
        if not batches:
            raise FileNotFoundError("No 'train_data_batch_*' files found in " + root)
    else:
        batches = ["val_data"]

    imgs_list, lbls_list = [], []
    for bname in batches:
        imgs, lbls = load_batch(os.path.join(root, bname))
        imgs_list.append(imgs)
        lbls_list.append(lbls)

    return np.concatenate(imgs_list), np.concatenate(lbls_list)


class ImageNet32(Dataset):  # type: ignore[misc]
    """PyTorch-friendly wrapper (optional)."""

    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None):
        self.images, self.labels = load_imagenet32(root, train)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):  # type: ignore[override]
        return self.images.shape[0]

    def __getitem__(self, idx):  # type: ignore[override]
        img, lbl = self.images[idx], int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        elif torch and not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl


def prepare_imagenet32(dataset_folder: Path, num_data: int, dataset_seed: int = 888, loss_type: str = 'mse'):
    """Prepare ImageNet32 dataset for training."""
    datafolder = dataset_folder / 'imagenet32'
    
    # Load train and test data
    train_images, train_labels = load_imagenet32(str(datafolder), train=True)
    test_images, test_labels = load_imagenet32(str(datafolder), train=False)
    
    train_size = num_data

    for partition in ['train', 'test']:
        if partition == 'train':
            partition_data = train_images
            partition_target = train_labels
            
            if num_data > 0 and num_data < len(partition_data):
                rng = np.random.default_rng(dataset_seed)
                idx = rng.choice(len(partition_data), train_size, replace=False)
                partition_data = partition_data[idx]
                partition_target = partition_target[idx]
        else:
            partition_data = test_images
            partition_target = test_labels

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        # ImageNet normalization values
        X = X - T.tensor([0.485, 0.456, 0.406])
        X = X / T.tensor([0.229, 0.224, 0.225])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        Y = Y - 1 # ImageNet labels are from 1 to 1000, not 0 to 999
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=1000).float()  # ImageNet has 1000 classes
        else:
            Y = Y.long()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test




def prepare_dataset(dataset: str, dataset_folder: Union[str, Path], num_data: int, classes: list, dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    dataset_folder = Path(dataset_folder)
    train_size = num_data
    if dataset == 'cifar10_2cls':
        # if loss_type == 'ce':
        #     raise NotImplementedError("Cross-entropy loss is not supported for 2-class CIFAR-10 dataset YET")
        return prepare_cifar10_2cls(dataset_folder, num_data, classes, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'cifar10':
        return prepare_cifar10(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'cifar10_ez':
        return prepare_cifar10_ez(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'svhn':
        return prepare_svhn(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'fmnist':
        return prepare_fmnist(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'imagenet32':
        return prepare_imagenet32(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
   
## NEW FUNCTION: Generate centroids, boundary points, x-outlier, y-outlier based on class selection

N_PROTOTYPE = 50
EXTRAPOLATION_FACTOR = 3.0


def _split_by_classes(
    X: T.Tensor,
    Y: T.Tensor,
    classes: tuple,
) -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
    """
    Convert labels to class indices (if needed) and return per-class tensors.
    Returns (X_class0, X_class1, idx_class0, idx_class1, class_labels)
    where idx tensors refer to positions in the original dataset.
    """
    if Y.ndim > 1:
        Y = Y.argmax(dim=1)

    class_labels = Y.to(torch.long).detach().cpu()
    X_cpu = X.detach().cpu()

    class_0, class_1 = classes
    mask_0 = class_labels == class_0
    mask_1 = class_labels == class_1

    if mask_0.sum() == 0 or mask_1.sum() == 0:
        raise ValueError(f"One of the classes {classes} has zero samples.")

    idx_0 = mask_0.nonzero(as_tuple=False).view(-1)
    idx_1 = mask_1.nonzero(as_tuple=False).view(-1)

    return X_cpu[mask_0], X_cpu[mask_1], idx_0, idx_1, class_labels


def _nearest_to_target(
    class_data_flat: T.Tensor,
    target_flat: T.Tensor,
    count: int,
) -> T.Tensor:
    """
    Return indices of the `count` samples in `class_data_flat`
    that are closest to `target_flat` (usually a centroid).
    """
    if class_data_flat.size(0) == 0:
        raise ValueError("class_data_flat must contain at least one sample.")
    count = max(1, min(count, class_data_flat.size(0)))
    distances = T.cdist(class_data_flat, target_flat).squeeze(1)
    _, idx = T.topk(distances, k=count, largest=False)
    return idx


def _assign_synthetic_to_real(
    synthetic_flat: T.Tensor,
    class_data_flat: T.Tensor,
    class_indices: T.Tensor,
) -> T.Tensor:
    """
    Map each synthetic feature vector to the closest real sample in class_data_flat.
    Ensures (best-effort) uniqueness of assignments.
    Returns indices relative to class_data_flat.
    """
    if synthetic_flat.ndim != 2 or class_data_flat.ndim != 2:
        raise ValueError("synthetic_flat and class_data_flat must be 2D tensors")

    assigned = []
    used = set()
    for syn_vec in synthetic_flat:
        distances = T.cdist(syn_vec.unsqueeze(0), class_data_flat).squeeze(0)
        sorted_candidates = torch.argsort(distances)
        chosen = None
        for candidate in sorted_candidates.tolist():
            global_idx = int(class_indices[candidate].item())
            if global_idx not in used:
                used.add(global_idx)
                chosen = candidate
                break
        if chosen is None:
            chosen = int(sorted_candidates[0].item())
        assigned.append(chosen)

    return torch.tensor(assigned, dtype=torch.long)

def generate_prototype_sets(
    X_train,
    Y_train,
    classes,
    n_prototype=None,
    prototype_frac=0.05,
    *,
    return_indices: bool = False,
):

    """
    Generates prototype sets: boundary, x_outlier, y_outlier.
    Works with image tensors [N, C, H, W] and one-hot labels.
    """
    if n_prototype is None:
        n_prototype = max(1, int(round(X_train.shape[0] * prototype_frac)))

    class_0, class_1 = classes
    X_0, X_1, idx_0, idx_1, class_labels = _split_by_classes(X_train, Y_train, classes)

    n0, n1 = X_0.size(0), X_1.size(0)

    # How many prototypes per class we can actually take
    k0 = min(n_prototype, n0)
    k1 = min(n_prototype, n1)

    # Centroids in image space
    centroid_0 = X_0.mean(dim=0, keepdim=True)  # [1, C, H, W]
    centroid_1 = X_1.mean(dim=0, keepdim=True)  # [1, C, H, W]

    # Flatten for distance computations
    X0_flat = X_0.view(n0, -1)
    X1_flat = X_1.view(n1, -1)
    c0_flat = centroid_0.view(1, -1)
    c1_flat = centroid_1.view(1, -1)

    v_diff_flat = c1_flat - c0_flat  # [1, D]

    # Combined view for boundary KNN selection
    combined_inputs = torch.cat([X_0, X_1], dim=0)
    combined_labels = torch.cat([
        T.full((n0,), class_0, dtype=class_labels.dtype),
        T.full((n1,), class_1, dtype=class_labels.dtype),
    ])
    combined_flat = combined_inputs.view(n0 + n1, -1)

    # ---------- 1. Boundary points (k-NN ambiguity) ----------
    boundary_local_indices = {class_0: [], class_1: []}

    if (n0 + n1) > 1:
        knn_neighbors = min(32, n0 + n1 - 1)
        if knn_neighbors >= 1:
            knn_results = identify_knn_outliers_by_neighbor_mix(
                combined_flat,
                combined_labels,
                k_neighbors=knn_neighbors,
                top_k_per_class=max(k0, k1),
                balance_target=0.5,
                chunk_size=min(1024, n0 + n1),
                normalize=True,
                return_neighbor_indices=False,
            )
            for class_value, entries in knn_results.get("outliers", {}).items():
                class_value = int(class_value)
                if class_value not in boundary_local_indices:
                    continue
                limit = k0 if class_value == class_0 else k1
                for entry in entries[:limit]:
                    dataset_idx = int(entry["dataset_index"])
                    if class_value == class_0:
                        local_idx = dataset_idx
                        if not (0 <= local_idx < n0):
                            continue
                    else:
                        local_idx = dataset_idx - n0
                        if not (0 <= local_idx < n1):
                            continue
                    boundary_local_indices[class_value].append(local_idx)

    # Fallback to centroid-based selection if needed
    dist_0_to_1 = T.cdist(X0_flat, c1_flat).squeeze(1)  # [n0]
    dist_1_to_0 = T.cdist(X1_flat, c0_flat).squeeze(1)  # [n1]
    _, idx_0_boundary_fallback = T.topk(dist_0_to_1, k=k0, largest=False)
    _, idx_1_boundary_fallback = T.topk(dist_1_to_0, k=k1, largest=False)

    def _ensure_boundary(class_value, desired_count, fallback_idx):
        existing = boundary_local_indices[class_value]
        seen = set(existing)
        if len(existing) < desired_count:
            for candidate in fallback_idx.tolist():
                if candidate not in seen:
                    existing.append(int(candidate))
                    seen.add(int(candidate))
                if len(existing) == desired_count:
                    break
        boundary_local_indices[class_value] = existing[:desired_count]

    _ensure_boundary(class_0, k0, idx_0_boundary_fallback)
    _ensure_boundary(class_1, k1, idx_1_boundary_fallback)

    idx_0_boundary = torch.tensor(boundary_local_indices[class_0], dtype=torch.long)
    idx_1_boundary = torch.tensor(boundary_local_indices[class_1], dtype=torch.long)
    boundary_idx_0_global = idx_0.index_select(0, idx_0_boundary)
    boundary_idx_1_global = idx_1.index_select(0, idx_1_boundary)
    boundary_global_idx = torch.cat([boundary_idx_0_global, boundary_idx_1_global], dim=0)

    X_boundary = T.cat([X_0[idx_0_boundary], X_1[idx_1_boundary]], dim=0)
    Y_boundary = T.cat([
        T.full((k0,), class_0, dtype=class_labels.dtype),
        T.full((k1,), class_1, dtype=class_labels.dtype),
    ])

    # ---------- 2. X-outliers (extrapolate along centroid diff) ----------
    dist_0_to_0 = T.cdist(X0_flat, c0_flat).squeeze(1)  # [n0]
    _, idx_0_near = T.topk(dist_0_to_0, k=k0, largest=False)

    X_seed = X0_flat[idx_0_near]                             # [k0, D]
    X_x_outlier_flat = X_seed - EXTRAPOLATION_FACTOR * v_diff_flat  # [k0, D]
    X_x_outlier = X_x_outlier_flat.view(k0, *X_0.shape[1:])  # back to [k0, C, H, W]

    Y_x_outlier = T.full((k0,), class_0, dtype=class_labels.dtype)

    # ---------- 3. Y-outliers (flip labels near centroids) ----------
    # C0 near its own centroid, relabeled as class_1
    X_y_outlier_0 = X_0[idx_0_near]
    Y_y_outlier_0 = T.full((k0,), class_1, dtype=class_labels.dtype)

    # C1 near its centroid, relabeled as class_0
    dist_1_to_1 = T.cdist(X1_flat, c1_flat).squeeze(1)  # [n1]
    _, idx_1_near = T.topk(dist_1_to_1, k=k1, largest=False)

    X_y_outlier_1 = X_1[idx_1_near]
    Y_y_outlier_1 = T.full((k1,), class_0, dtype=class_labels.dtype)

    X_y_outlier = T.cat([X_y_outlier_0, X_y_outlier_1], dim=0)
    Y_y_outlier = T.cat([Y_y_outlier_0, Y_y_outlier_1], dim=0)

    
    # ---------- 4. Inliers (close to centroids) ----------
    # Class 0 Inliers (Features from C0, Label = C0)
    X_inlier_0 = X_0[idx_0_near]
    Y_inlier_0 = T.full((k0,), class_0, dtype=class_labels.dtype)

    # Class 1 Inliers (Features from C1, Label = C1)
    X_inlier_1 = X_1[idx_1_near]
    Y_inlier_1 = T.full((k1,), class_1, dtype=class_labels.dtype)
    
    # Concatenate to form the final Inlier set
    X_inlier = T.cat([X_inlier_0, X_inlier_1], dim=0)
    Y_inlier = T.cat([Y_inlier_0, Y_inlier_1], dim=0)
    inlier_idx_0_global = idx_0.index_select(0, idx_0_near)
    inlier_idx_1_global = idx_1.index_select(0, idx_1_near)
    inlier_global_idx = torch.cat([inlier_idx_0_global, inlier_idx_1_global], dim=0)

    prototypes = {
        'boundary': (X_boundary, Y_boundary),
        'x_outlier': (X_x_outlier, Y_x_outlier),
        'y_outlier': (X_y_outlier, Y_y_outlier),
        'inliers': (X_inlier, Y_inlier),
    }
    if return_indices:
        return prototypes, {
            'boundary': boundary_global_idx,
            'inliers': inlier_global_idx,
        }
    return prototypes


def _select_indices_by_class(
    labels: T.Tensor,
    classes: tuple,
    count_per_class: int,
) -> T.Tensor:
    if count_per_class < 1:
        raise ValueError("count_per_class must be >= 1")

    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    labels = labels.to(dtype=torch.long)

    selected = []
    for class_id in classes:
        class_idx = (labels == class_id).nonzero(as_tuple=False).view(-1)
        if class_idx.numel() == 0:
            continue
        take = min(count_per_class, class_idx.numel())
        selected.append(class_idx[:take])

    if not selected:
        return torch.empty((0,), dtype=torch.long)

    return torch.cat(selected, dim=0)


def trim_prototype_sets(
    prototypes: Dict[str, Tuple[T.Tensor, T.Tensor]],
    classes: tuple,
    counts_by_subset: Optional[Dict[str, int]] = None,
    indices: Optional[Dict[str, T.Tensor]] = None,
) -> Tuple[Dict[str, Tuple[T.Tensor, T.Tensor]], Optional[Dict[str, T.Tensor]]]:
    if not counts_by_subset:
        return prototypes, indices

    trimmed = {}
    trimmed_indices = {} if indices is not None else None

    for name, (X, Y) in prototypes.items():
        count = counts_by_subset.get(name)
        if count is None:
            trimmed[name] = (X, Y)
            if trimmed_indices is not None and indices and name in indices:
                trimmed_indices[name] = indices[name]
            continue

        selected = _select_indices_by_class(Y, classes, count)
        selected = selected.to(device=X.device)
        trimmed[name] = (X.index_select(0, selected), Y.index_select(0, selected))

        if trimmed_indices is not None and indices and name in indices:
            idx_tensor = indices[name]
            if not torch.is_tensor(idx_tensor):
                idx_tensor = torch.tensor(idx_tensor, dtype=torch.long)
            trimmed_indices[name] = idx_tensor.index_select(0, selected.to(idx_tensor.device))

    return trimmed, trimmed_indices


def generate_feature_space_prototype_sets(
    net: Optional[nn.Module],
    Y_train: T.Tensor,
    classes: tuple,
    *,
    inputs: Optional[T.Tensor] = None,
    precomputed_features: Optional[T.Tensor] = None,
    original_inputs: Optional[T.Tensor] = None,
    feature_batch_size: int = 512,
    flatten_outputs: bool = True,
    normalize_features: bool = True,
    k_neighbors: int = 32,
    prototypes_per_class: int = N_PROTOTYPE,
    extrapolation_factor: float = EXTRAPOLATION_FACTOR,
) -> Tuple[Dict[str, Tuple[T.Tensor, T.Tensor]], Dict[str, dict]]:
    """
    Generate centroid/boundary/outlier prototype sets in feature space and map them
    back to raw inputs for downstream curvature tracking.

    Returns:
        (prototypes, metadata)

        prototypes: dict mapping subset names to (inputs, labels) tensors.
        metadata: dict with centroids/indices/feature tensors useful for logging.
    """
    if precomputed_features is None:
        if net is None or inputs is None:
            raise ValueError("Either `precomputed_features` or both `net` and `inputs` must be provided.")
        features = extract_feature_matrix(
            net=net,
            inputs=inputs,
            batch_size=feature_batch_size,
            flatten_outputs=flatten_outputs,
        )
    else:
        features = precomputed_features

    if original_inputs is None:
        if inputs is None:
            raise ValueError("original_inputs must be provided when inputs is None.")
        original_inputs = inputs

    features = features.detach().to(torch.float32)
    if normalize_features:
        features = torch.nn.functional.normalize(features, dim=1)

    features_cpu = features.cpu()
    Y_cpu = Y_train.detach().cpu()
    X_ref = original_inputs.detach().cpu()

    X_0, X_1, idx_0, idx_1, class_labels = _split_by_classes(features_cpu, Y_cpu, classes)

    n0, n1 = X_0.size(0), X_1.size(0)
    k0 = min(prototypes_per_class, n0)
    k1 = min(prototypes_per_class, n1)

    centroid_0 = X_0.mean(dim=0, keepdim=True)
    centroid_1 = X_1.mean(dim=0, keepdim=True)

    X0_flat = X_0.view(n0, -1)
    X1_flat = X_1.view(n1, -1)
    c0_flat = centroid_0.view(1, -1)
    c1_flat = centroid_1.view(1, -1)
    v_diff_flat = c1_flat - c0_flat

    idx_0_near = _nearest_to_target(X0_flat, c0_flat, k0)
    idx_1_near = _nearest_to_target(X1_flat, c1_flat, k1)

    dist_0_to_1 = T.cdist(X0_flat, c1_flat).squeeze(1)
    dist_1_to_0 = T.cdist(X1_flat, c0_flat).squeeze(1)
    _, idx_0_boundary_fallback = T.topk(dist_0_to_1, k=k0, largest=False)
    _, idx_1_boundary_fallback = T.topk(dist_1_to_0, k=k1, largest=False)

    boundary_indices = {classes[0]: [], classes[1]: []}
    knn_results = None
    if k_neighbors is not None and k_neighbors >= 1 and features_cpu.size(0) > k_neighbors:
        knn_results = identify_knn_outliers_by_neighbor_mix(
            features_cpu,
            class_labels,
            k_neighbors=k_neighbors,
            top_k_per_class=max(k0, k1),
            normalize=False,
        )

    if knn_results is not None:
        for class_value, entries in knn_results.get("outliers", {}).items():
            class_value = int(class_value)
            if class_value in boundary_indices:
                boundary_indices[class_value].extend(
                    entry["dataset_index"] for entry in entries
                )

    def _fill_boundary(class_value, desired_count, fallback_local_idx, dataset_indices):
        existing = boundary_indices[class_value]
        seen = set(existing)
        if len(existing) > desired_count:
            del existing[desired_count:]
        elif len(existing) < desired_count:
            for rel_idx in fallback_local_idx.tolist():
                dataset_idx = int(dataset_indices[rel_idx].item())
                if dataset_idx in seen:
                    continue
                existing.append(dataset_idx)
                seen.add(dataset_idx)
                if len(existing) == desired_count:
                    break
        boundary_indices[class_value] = existing[:desired_count]

    _fill_boundary(classes[0], k0, idx_0_boundary_fallback, idx_0)
    _fill_boundary(classes[1], k1, idx_1_boundary_fallback, idx_1)

    boundary_idx_0 = torch.tensor(boundary_indices[classes[0]], dtype=torch.long)
    boundary_idx_1 = torch.tensor(boundary_indices[classes[1]], dtype=torch.long)
    boundary_global_idx = torch.cat([boundary_idx_0, boundary_idx_1], dim=0)

    Y_boundary = torch.cat([
        T.full((boundary_idx_0.numel(),), classes[0], dtype=class_labels.dtype),
        T.full((boundary_idx_1.numel(),), classes[1], dtype=class_labels.dtype),
    ])

    X_seed = X0_flat[idx_0_near]
    X_x_outlier_flat = X_seed - extrapolation_factor * v_diff_flat
    X_x_outlier = X_x_outlier_flat.view(k0, -1)

    assigned_local = _assign_synthetic_to_real(X_x_outlier_flat, X0_flat, idx_0)
    x_outlier_global_idx = idx_0[assigned_local]
    Y_x_outlier = T.full((x_outlier_global_idx.numel(),), classes[0], dtype=class_labels.dtype)

    y0_global_idx = idx_0[idx_0_near]
    y1_global_idx = idx_1[idx_1_near]
    X_y_outlier_idx = torch.cat([y0_global_idx, y1_global_idx], dim=0)
    Y_y_outlier = torch.cat([
        T.full((k0,), classes[1], dtype=class_labels.dtype),
        T.full((k1,), classes[0], dtype=class_labels.dtype),
    ])

    inlier_idx = torch.cat([y0_global_idx, y1_global_idx], dim=0)
    Y_inlier = torch.cat([
        T.full((k0,), classes[0], dtype=class_labels.dtype),
        T.full((k1,), classes[1], dtype=class_labels.dtype),
    ])

    def _gather_inputs(indices: torch.Tensor) -> T.Tensor:
        return X_ref.index_select(0, indices)

    prototypes = {
        'feature_boundary': (_gather_inputs(boundary_global_idx), Y_boundary),
        'feature_x_outlier': (_gather_inputs(x_outlier_global_idx), Y_x_outlier),
        'feature_y_outlier': (_gather_inputs(X_y_outlier_idx), Y_y_outlier),
        'feature_inliers': (_gather_inputs(inlier_idx), Y_inlier),
    }

    metadata = {
        'classes': [int(classes[0]), int(classes[1])],
        'centroids': {
            str(int(classes[0])): centroid_0.squeeze(0).cpu(),
            str(int(classes[1])): centroid_1.squeeze(0).cpu(),
        },
        'indices': {
            'feature_boundary': boundary_global_idx.cpu(),
            'feature_x_outlier': x_outlier_global_idx.cpu(),
            'feature_y_outlier': X_y_outlier_idx.cpu(),
            'feature_inliers': inlier_idx.cpu(),
        },
        'features': {
            'feature_boundary': features_cpu.index_select(0, boundary_global_idx),
            'feature_x_outlier': features_cpu.index_select(0, x_outlier_global_idx),
            'feature_y_outlier': features_cpu.index_select(0, X_y_outlier_idx),
            'feature_inliers': features_cpu.index_select(0, inlier_idx),
            'feature_x_outlier_synthetic': X_x_outlier.cpu(),
        },
    }

    return prototypes, metadata
